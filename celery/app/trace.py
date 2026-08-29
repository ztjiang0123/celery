"""Trace task execution.

This module defines how the task execution is traced:
errors are recorded, handlers are applied and so on.
"""
import logging
import os
import sys
import time
from collections import namedtuple
from warnings import warn

from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu.exceptions import EncodeError
from kombu.serialization import loads as loads_message
from kombu.serialization import prepare_accept_content
from kombu.utils.encoding import safe_repr, safe_str

from celery import current_app, group, signals, states
from celery._state import _task_stack
from celery.app.task import Context
from celery.app.task import Task as BaseTask
from celery.exceptions import BackendGetMetaError, Ignore, InvalidTaskError, Reject, Retry
from celery.result import AsyncResult
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import mro_lookup
from celery.utils.saferepr import saferepr
from celery.utils.serialization import get_pickleable_etype, get_pickleable_exception, get_pickled_exception

# ## ---
# This is the heart of the worker, the inner loop so to speak.
# It used to be split up into nice little classes and methods,
# but in the end it only resulted in bad performance and horrible tracebacks,
# so instead we now use one closure per task class.

# pylint: disable=redefined-outer-name
# We cache globals and attribute lookups, so disable this warning.
# pylint: disable=broad-except
# We know what we're doing...


__all__ = (
    'TraceInfo', 'build_tracer', 'trace_task',
    'setup_worker_optimizations', 'reset_worker_optimizations',
)

from celery.worker.state import successful_requests

logger = get_logger(__name__)

#: Format string used to log task receipt.
LOG_RECEIVED = """\
Task %(name)s[%(id)s] received\
"""

#: Format string used to log task success.
LOG_SUCCESS = """\
Task %(name)s[%(id)s] succeeded in %(runtime)ss: %(return_value)s\
"""

#: Format string used to log task failure.
LOG_FAILURE = """\
Task %(name)s[%(id)s] %(description)s: %(exc)s\
"""

#: Format string used to log task internal error.
LOG_INTERNAL_ERROR = """\
Task %(name)s[%(id)s] %(description)s: %(exc)s\
"""

#: Format string used to log task ignored.
LOG_IGNORED = """\
Task %(name)s[%(id)s] %(description)s\
"""

#: Format string used to log task rejected.
LOG_REJECTED = """\
Task %(name)s[%(id)s] %(exc)s\
"""

#: Format string used to log task retry.
LOG_RETRY = """\
Task %(name)s[%(id)s] retry: %(exc)s\
"""

log_policy_t = namedtuple(
    'log_policy_t',
    ('format', 'description', 'severity', 'traceback', 'mail'),
)

log_policy_reject = log_policy_t(LOG_REJECTED, 'rejected', logging.WARN, 1, 1)
log_policy_ignore = log_policy_t(LOG_IGNORED, 'ignored', logging.INFO, 0, 0)
log_policy_internal = log_policy_t(
    LOG_INTERNAL_ERROR, 'INTERNAL ERROR', logging.CRITICAL, 1, 1,
)
log_policy_expected = log_policy_t(
    LOG_FAILURE, 'raised expected', logging.INFO, 0, 0,
)
log_policy_unexpected = log_policy_t(
    LOG_FAILURE, 'raised unexpected', logging.ERROR, 1, 1,
)

send_prerun = signals.task_prerun.send
send_postrun = signals.task_postrun.send
send_success = signals.task_success.send
STARTED = states.STARTED
SUCCESS = states.SUCCESS
IGNORED = states.IGNORED
REJECTED = states.REJECTED
RETRY = states.RETRY
FAILURE = states.FAILURE
EXCEPTION_STATES = states.EXCEPTION_STATES
IGNORE_STATES = frozenset({IGNORED, RETRY, REJECTED})

#: set by :func:`setup_worker_optimizations`
_localized = []
_patched = {}

trace_ok_t = namedtuple('trace_ok_t', ('retval', 'info', 'runtime', 'retstr'))

#: Mutable-by-``_replace`` bundle of the six variables threaded through the
#: task-tracing steps.  Field order matches the historical
#: ``R, I, T, Rstr, retval, state`` locals of ``trace_task``.
_TraceVars = namedtuple('_TraceVars', ('R', 'I', 'T', 'Rstr', 'retval', 'state'))


def info(fmt, context):
    """Log 'fmt % context' with severity 'INFO'.

    'context' is also passed in extra with key 'data' for custom handlers.
    """
    logger.info(fmt, context, extra={'data': context})


def task_has_custom(task, attr):
    """Return true if the task overrides ``attr``."""
    return mro_lookup(task.__class__, attr, stop={BaseTask, object},
                      monkey_patched=['celery.app.task'])


def get_log_policy(task, einfo, exc):
    if isinstance(exc, Reject):
        return log_policy_reject
    elif isinstance(exc, Ignore):
        return log_policy_ignore
    elif einfo.internal:
        return log_policy_internal
    else:
        if task.throws and isinstance(exc, task.throws):
            return log_policy_expected
        return log_policy_unexpected


def get_task_name(request, default):
    """Use 'shadow' in request for the task name if applicable."""
    # request.shadow could be None or an empty string.
    # If so, we should use default.
    return getattr(request, 'shadow', None) or default


def get_actual_ignore_result(task, req):
    """Return the effective ignore_result, with request overriding task.

    If req provides an explicit ignore_result, that value is used;
    otherwise task.ignore_result is returned.
    """
    if req is None:
        return task.ignore_result

    actual = getattr(req, 'ignore_result', None)

    # Context defines `ignore_result = False` at class level (see Context
    # in celery/app/task.py). getattr() above would return the class default
    # (False) even when the request never set it explicitly, making it
    # impossible to distinguish "override=False" from "not set". We check
    # __dict__ to detect only instance-level (i.e., explicitly set) values.
    if isinstance(req, Context) and 'ignore_result' not in req.__dict__:
        actual = None

    return actual if actual is not None else task.ignore_result


class TraceInfo:
    """Information about task execution."""

    __slots__ = ('state', 'retval')

    def __init__(self, state, retval=None):
        self.state = state
        self.retval = retval

    def handle_error_state(self, task, req,
                           eager=False, call_errbacks=True):
        ignore_result = get_actual_ignore_result(task, req)

        if ignore_result:
            store_errors = task.store_errors_even_if_ignored
        elif eager and task.store_eager_result:
            store_errors = True
        else:
            store_errors = not eager

        return {
            RETRY: self.handle_retry,
            FAILURE: self.handle_failure,
        }[self.state](task, req,
                      store_errors=store_errors,
                      call_errbacks=call_errbacks)

    def handle_reject(self, task, req, **kwargs):
        self._log_error(task, req, ExceptionInfo())

    def handle_ignore(self, task, req, **kwargs):
        self._log_error(task, req, ExceptionInfo())

    def handle_retry(self, task, req, store_errors=True, **kwargs):
        """Handle retry exception."""
        # the exception raised is the Retry semi-predicate,
        # and it's exc' attribute is the original exception raised (if any).
        type_, _, tb = sys.exc_info()
        einfo = None
        try:
            reason = self.retval
            einfo = ExceptionInfo((type_, reason, tb))
            if store_errors:
                task.backend.mark_as_retry(
                    req.id, reason.exc, einfo.traceback, request=req,
                )
            task.on_retry(reason.exc, req.id, req.args, req.kwargs, einfo)
            signals.task_retry.send(sender=task, request=req,
                                    reason=reason, einfo=einfo)
            info(LOG_RETRY, {
                'id': req.id,
                'name': get_task_name(req, task.name),
                'exc': str(reason),
            })
            # MEMORY LEAK FIX: Clear traceback frames to prevent memory retention (Issue #8882)
            traceback_clear(einfo.exception)
            return einfo
        finally:
            # MEMORY LEAK FIX: Clean up direct traceback reference to prevent
            # retention of frame objects and their local variables (Issue #8882)
            if tb is not None:
                del tb

    def handle_failure(self, task, req, store_errors=True, call_errbacks=True):
        """Handle exception."""
        orig_exc = self.retval
        tb_ref = None

        try:
            exc = get_pickleable_exception(orig_exc)
            if exc.__traceback__ is None:
                # `get_pickleable_exception` may have created a new exception without
                # a traceback.
                _, _, tb_ref = sys.exc_info()
                exc.__traceback__ = tb_ref

            exc_type = get_pickleable_etype(type(orig_exc))

            # make sure we only send pickleable exceptions back to parent.
            einfo = ExceptionInfo(exc_info=(exc_type, exc, exc.__traceback__))

            task.backend.mark_as_failure(
                req.id, exc, einfo.traceback,
                request=req, store_result=store_errors,
                call_errbacks=call_errbacks,
            )

            task.on_failure(exc, req.id, req.args, req.kwargs, einfo)
            signals.task_failure.send(sender=task, task_id=req.id,
                                      exception=exc, args=req.args,
                                      kwargs=req.kwargs,
                                      traceback=exc.__traceback__,
                                      einfo=einfo)
            self._log_error(task, req, einfo)
            # MEMORY LEAK FIX: Clear traceback frames to prevent memory retention (Issue #8882)
            traceback_clear(exc)
            # Note: We return einfo, so we can't clean it up here
            # The calling function is responsible for cleanup
            return einfo
        finally:
            # MEMORY LEAK FIX: Clean up any direct traceback references we may have created
            # to prevent retention of frame objects and their local variables (Issue #8882)
            if tb_ref is not None:
                del tb_ref

    def _log_error(self, task, req, einfo):
        eobj = einfo.exception = get_pickled_exception(einfo.exception)
        if isinstance(eobj, ExceptionWithTraceback):
            eobj = einfo.exception = eobj.exc
        exception, traceback, exc_info, sargs, skwargs = (
            safe_repr(eobj),
            safe_str(einfo.traceback),
            einfo.exc_info,
            req.get('argsrepr') or safe_repr(req.args),
            req.get('kwargsrepr') or safe_repr(req.kwargs),
        )
        policy = get_log_policy(task, einfo, eobj)

        context = {
            'hostname': req.hostname,
            'id': req.id,
            'name': get_task_name(req, task.name),
            'exc': exception,
            'traceback': traceback,
            'args': sargs,
            'kwargs': skwargs,
            'description': policy.description,
            'internal': einfo.internal,
        }

        logger.log(policy.severity, policy.format.strip(), context,
                   exc_info=exc_info if policy.traceback else None,
                   extra={'data': context})


def traceback_clear(exc=None):
    """Clear traceback frames to prevent memory leaks.

    MEMORY LEAK FIX: This function helps break reference cycles between
    traceback objects and frame objects that can prevent garbage collection.
    Clearing frames releases local variables that may be holding large objects.
    """
    # Cleared Tb, but einfo still has a reference to Traceback.
    # exc cleans up the Traceback at the last moment that can be revealed.
    tb = None
    if exc is not None:
        if hasattr(exc, '__traceback__'):
            tb = exc.__traceback__
        else:
            _, _, tb = sys.exc_info()
    else:
        _, _, tb = sys.exc_info()

    while tb is not None:
        try:
            # MEMORY LEAK FIX: tb.tb_frame.clear() clears ALL frame data including
            # local variables, which is more efficient than accessing f_locals separately.
            # Removed redundant tb.tb_frame.f_locals access that was creating unnecessary references.
            tb.tb_frame.clear()
        except RuntimeError:
            # Ignore the exception raised if the frame is still executing.
            pass
        tb = tb.tb_next


class _TraceContext:
    """Immutable-ish bundle of the state ``build_tracer`` prepares once.

    ``build_tracer`` historically closed over ~30 locals shared by a stack
    of nested helper functions.  Collecting them here lets those helpers
    live at module scope (flat, individually readable) instead of nesting
    inside ``build_tracer``, while the per-task hot path still reads every
    value with a single attribute access.
    """

    __slots__ = (
        'name', 'task', 'fun', 'app', 'Info', 'eager', 'propagate',
        'monotonic', 'trace_ok_t', 'IGNORE_STATES', 'signature',
        'hostname', 'pid', 'loader_task_init', 'loader_cleanup',
        'task_before_start', 'task_on_success', 'task_after_return',
        'push_request', 'pop_request', 'push_task', 'pop_task',
        'prerun_receivers', 'postrun_receivers', 'success_receivers',
        'deduplicate_successful_tasks', 'successful_requests',
        'inherit_parent_priority', 'resultrepr_maxsize', '_does_info',
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _TraceRun:
    """Per-invocation state threaded through the task-tracing helpers.

    ``build_tracer``'s helpers each needed most of the same call-scoped
    values (uuid, args, the request, ids, timing, ...).  Bundling them here
    lets every helper take just ``(ctx, run)`` instead of a long parameter
    list, while ``tvars`` carries the six mutable trace variables.
    """

    __slots__ = (
        'uuid', 'args', 'kwargs', 'task_request', 'root_id', 'task_priority',
        'time_start', 'publish_result', 'track_started', 'tvars',
    )

    def __init__(self, uuid, args, kwargs, time_start):
        self.uuid = uuid
        self.args = args
        self.kwargs = kwargs
        self.time_start = time_start
        self.task_request = None
        self.root_id = None
        self.task_priority = None
        self.publish_result = False
        self.track_started = False
        self.tvars = _TraceVars(None, None, None, None, None, None)


#: Where a dispatched signature should attach itself in the task tree.
_DispatchTarget = namedtuple('_DispatchTarget', ('parent_id', 'root_id', 'priority'))


def _trace_on_error(ctx, request, exc, retry=False):
    """Handle an errored task, returning ``(I, R, state, retval)``.

    ``retry`` selects the RETRY state and suppresses errbacks, matching the
    historical ``on_error(..., RETRY, call_errbacks=False)`` call.
    """
    if ctx.propagate:
        raise
    state = RETRY if retry else FAILURE
    I = ctx.Info(state, exc)
    R = I.handle_error_state(
        ctx.task, request, eager=ctx.eager, call_errbacks=not retry,
    )
    return I, R, I.state, I.retval


def _dispatch_multiple_callbacks(ctx, retval, callbacks, target):
    """Apply link callbacks when there is more than one.

    Groups are applied individually (so their trail is stored once); the
    remaining plain signatures are applied together as one group.
    """
    signature = ctx.signature
    sigs, groups = [], []
    for sig in callbacks:
        sig = signature(sig, app=ctx.app)
        if isinstance(sig, group):
            groups.append(sig)
        else:
            sigs.append(sig)
    for group_ in groups:
        group_.apply_async(
            (retval,),
            parent_id=target.parent_id, root_id=target.root_id,
            priority=target.priority,
        )
    if sigs:
        group(sigs, app=ctx.app).apply_async(
            (retval,),
            parent_id=target.parent_id, root_id=target.root_id,
            priority=target.priority,
        )


def _dispatch_callbacks_and_chain(ctx, retval, callbacks, chain, target):
    """Dispatch callbacks and chain for a completed task.

    Dispatches link callbacks and then the next chain step.  Does NOT fire
    task lifecycle signals (on_success, task_postrun) or call mark_as_done —
    callers handle those separately.

    Note: dispatch is not atomic.  If callbacks succeed but the chain step
    fails (or vice-versa), a Reject + redeliver may re-dispatch the already-
    sent callbacks.  This is acceptable under Celery's at-least-once
    delivery model.
    """
    if callbacks and len(callbacks) > 1:
        _dispatch_multiple_callbacks(ctx, retval, callbacks, target)
    elif callbacks:
        ctx.signature(callbacks[0], app=ctx.app).apply_async(
            (retval,),
            parent_id=target.parent_id, root_id=target.root_id,
            priority=target.priority,
        )
    if chain:
        _chsig = ctx.signature(chain[-1], app=ctx.app)
        _chsig.apply_async(
            (retval,), chain=chain[:-1],
            parent_id=target.parent_id, root_id=target.root_id,
            priority=target.priority,
        )


def _redispatch_deduplicated_result(ctx, task_request, uuid, r):
    """Re-dispatch callbacks/chain for an already-successful redelivery.

    The task itself is not re-run; we only make sure the follow-up callbacks
    and chain steps fire (unless they already did) and record the request as
    handled.
    """
    info(LOG_IGNORED, {
        'id': task_request.id,
        'name': get_task_name(task_request, ctx.name),
        'description': 'Task already completed successfully.'
    })
    root_id = task_request.root_id or uuid
    priority = task_request.delivery_info.get('priority') if \
        ctx.inherit_parent_priority else None
    try:
        meta = r._get_task_meta()
        stored_retval = meta.get('result')
        # Children are populated by mark_as_done on the original execution.
        # If present, callbacks were already dispatched -- skip to avoid
        # duplicates.  Requires the backend to persist extended result
        # metadata (result_extended=True).
        children = meta.get('children')
        callbacks = task_request.callbacks
        chain = task_request.chain
        if (callbacks or chain) and not children:
            _dispatch_callbacks_and_chain(
                ctx, stored_retval, callbacks, chain,
                _DispatchTarget(uuid, root_id, priority),
            )
        ctx.successful_requests.add(task_request.id)
    except MemoryError:
        raise
    except Exception as exc:
        # Permanent failures (malformed signature, etc.) will requeue
        # indefinitely.  Broker-level dead-letter / max-delivery-count
        # policies are the intended circuit-breaker.
        logger.error(
            'Failed to dispatch chain/callbacks for deduplicated task %s',
            task_request.id, exc_info=True,
        )
        raise Reject(exc, requeue=True)


def _is_duplicate_success(ctx, task_request, uuid):
    """Return True if this redelivered request already ran successfully.

    Handles re-dispatching callbacks for the already-completed task as a
    side effect; the caller should stop tracing when this returns True.
    """
    redelivered = (task_request.delivery_info
                   and task_request.delivery_info.get('redelivered', False))
    if not (ctx.deduplicate_successful_tasks and redelivered):
        return False

    if task_request.id in ctx.successful_requests:
        return True

    r = AsyncResult(task_request.id, app=ctx.app)
    try:
        state = r.state
    except BackendGetMetaError:
        return False

    if state != SUCCESS:
        return False

    _redispatch_deduplicated_result(ctx, task_request, uuid, r)
    return True


def _on_task_success(ctx, run):
    """Handle the success path: dispatch callbacks, store, report.

    Mirrors the historical ``else`` branch of the TRACE block and returns
    the (possibly updated) ``_TraceVars`` tuple.
    """
    R, I, T, Rstr, retval, state = run.tvars
    uuid, task_request = run.uuid, run.task_request
    task = ctx.task
    try:
        # callback tasks must be applied before the result is stored, so
        # that result.children is populated.

        # groups are called inline and will store trail separately, so need
        # to call them separately so that the trail's not added multiple
        # times :( (Issue #1936)
        _dispatch_callbacks_and_chain(
            ctx, retval, task.request.callbacks, task_request.chain,
            _DispatchTarget(uuid, run.root_id, run.task_priority),
        )
        task.backend.mark_as_done(uuid, retval, task_request, run.publish_result)
    except EncodeError as exc:
        I, R, state, retval = _trace_on_error(ctx, task_request, exc)
        # MEMORY LEAK FIX: Clear traceback frames to prevent memory retention (Issue #8882)
        traceback_clear(exc)
    else:
        Rstr = saferepr(R, ctx.resultrepr_maxsize)
        T = ctx.monotonic() - run.time_start
        if ctx.task_on_success:
            ctx.task_on_success(retval, uuid, run.args, run.kwargs)
        if ctx.success_receivers:
            send_success(sender=task, result=retval)
        if ctx._does_info:
            info(LOG_SUCCESS, {
                'id': uuid,
                'name': get_task_name(task_request, ctx.name),
                'return_value': Rstr,
                'runtime': T,
                'args': task_request.get('argsrepr') or safe_repr(run.args),
                'kwargs': task_request.get('kwargsrepr') or safe_repr(run.kwargs),
            })
    return _TraceVars(R, I, T, Rstr, retval, state)


def _run_and_classify(ctx, run):
    """Run the task body and classify its outcome into trace variables.

    Corresponds to the historical TRACE ``try/except/else`` block: invokes
    ``before_start`` and the task, maps each terminal exception type to its
    state, and dispatches the success path.  Returns the updated
    ``_TraceVars`` tuple.
    """
    R, I, T, Rstr, retval, state = run.tvars
    uuid, args, kwargs = run.uuid, run.args, run.kwargs
    task_request = run.task_request
    task, Info = ctx.task, ctx.Info
    try:
        if ctx.task_before_start:
            ctx.task_before_start(uuid, args, kwargs)

        R = retval = ctx.fun(*args, **kwargs)
        state = SUCCESS
    except Reject as exc:
        I, R = Info(REJECTED, exc), ExceptionInfo(internal=True)
        state, retval = I.state, I.retval
        I.handle_reject(task, task_request)
        # MEMORY LEAK FIX: Clear traceback frames to prevent memory retention (Issue #8882)
        traceback_clear(exc)
    except Ignore as exc:
        I, R = Info(IGNORED, exc), ExceptionInfo(internal=True)
        state, retval = I.state, I.retval
        I.handle_ignore(task, task_request)
        # MEMORY LEAK FIX: Clear traceback frames to prevent memory retention (Issue #8882)
        traceback_clear(exc)
    except Retry as exc:
        I, R, state, retval = _trace_on_error(ctx, task_request, exc, retry=True)
        # MEMORY LEAK FIX: Clear traceback frames to prevent memory retention (Issue #8882)
        traceback_clear(exc)
    except Exception as exc:
        I, R, state, retval = _trace_on_error(ctx, task_request, exc)
        # MEMORY LEAK FIX: Clear traceback frames to prevent memory retention (Issue #8882)
        traceback_clear(exc)
    except BaseException:
        raise
    else:
        run.tvars = _TraceVars(R, I, T, Rstr, retval, state)
        return _on_task_success(ctx, run)
    return _TraceVars(R, I, T, Rstr, retval, state)


def _run_process_cleanup(ctx):
    """Run backend/loader process cleanup unless tracing eagerly."""
    if ctx.eager:
        return
    try:
        ctx.task.backend.process_cleanup()
        ctx.loader_cleanup()
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        logger.error('Process cleanup failed: %r', exc, exc_info=True)


def _resolve_publish_result(ctx, task, ignore_result):
    """Decide whether the result should be published for this run."""
    # #6476
    if ctx.eager and not ignore_result and task.store_eager_result:
        return True
    return not ctx.eager and not ignore_result


def _trace_task_lifecycle(ctx, run):
    """Run PRE / TRACE / POST inside the request-stack scope.

    Returns the updated ``_TraceVars`` tuple.  Assumes the task and request
    have already been pushed; the caller owns popping them.
    """
    uuid, args, kwargs = run.uuid, run.args, run.kwargs
    task_request = run.task_request
    task = ctx.task
    tvars = run.tvars
    try:
        # -*- PRE -*-
        if ctx.prerun_receivers:
            send_prerun(sender=task, task_id=uuid, task=task,
                        args=args, kwargs=kwargs)
        ctx.loader_task_init(uuid, task)
        if run.track_started:
            task.backend.store_result(
                uuid, {'pid': ctx.pid, 'hostname': ctx.hostname}, STARTED,
                request=task_request,
            )

        # -*- TRACE -*-
        tvars = _run_and_classify(ctx, run)

        # -* POST *-
        state, retval = tvars.state, tvars.retval
        if state not in ctx.IGNORE_STATES and ctx.task_after_return:
            ctx.task_after_return(state, retval, uuid, args, kwargs, None)
        return tvars
    finally:
        try:
            if ctx.postrun_receivers:
                send_postrun(sender=task, task_id=uuid, task=task,
                             args=args, kwargs=kwargs,
                             retval=tvars.retval, state=tvars.state)
        finally:
            ctx.pop_task()
            ctx.pop_request()
            _run_process_cleanup(ctx)


def _trace_task(ctx, uuid, args, kwargs, request=None):
    # R      - is the possibly prepared return value.
    # I      - is the Info object.
    # T      - runtime
    # Rstr   - textual representation of return value
    # retval - is the always unmodified return value.
    # state  - is the resulting task state.
    task = ctx.task
    run = _TraceRun(uuid, args, kwargs, ctx.monotonic())
    try:
        try:
            kwargs.items
        except AttributeError:
            raise InvalidTaskError('Task keyword arguments is not a mapping')

        run.task_request = task_request = Context(
            request or {}, args=args, called_directly=False, kwargs=kwargs)

        ignore_result = get_actual_ignore_result(task, task_request)
        run.track_started = not ctx.eager and (task.track_started and not ignore_result)
        run.publish_result = _resolve_publish_result(ctx, task, ignore_result)

        if _is_duplicate_success(ctx, task_request, uuid):
            tvars = run.tvars
            return ctx.trace_ok_t(tvars.R, tvars.I, tvars.T, tvars.Rstr)

        ctx.push_task(task)
        run.root_id = task_request.root_id or uuid
        run.task_priority = task_request.delivery_info.get('priority') if \
            ctx.inherit_parent_priority else None
        ctx.push_request(task_request)
        run.tvars = _trace_task_lifecycle(ctx, run)
    except MemoryError:
        raise
    except Reject:
        raise
    except Exception as exc:
        _signal_internal_error(task, uuid, args, kwargs, request, exc)
        if ctx.eager:
            raise
        R = report_internal_error(task, exc)
        I = run.tvars.I
        if run.task_request is not None:
            I, _, _, _ = _trace_on_error(ctx, run.task_request, exc)
        run.tvars = run.tvars._replace(R=R, I=I)
    tvars = run.tvars
    return ctx.trace_ok_t(tvars.R, tvars.I, tvars.T, tvars.Rstr)


def build_tracer(name, task, loader=None, hostname=None, store_errors=True,
                 Info=TraceInfo, eager=False, propagate=False, app=None,
                 monotonic=time.monotonic, trace_ok_t=trace_ok_t,
                 IGNORE_STATES=IGNORE_STATES):
    """Return a function that traces task execution.

    Catches all exceptions and updates result backend with the
    state and result.

    If the call was successful, it saves the result to the task result
    backend, and sets the task status to `"SUCCESS"`.

    If the call raises :exc:`~@Retry`, it extracts
    the original exception, uses that as the result and sets the task state
    to `"RETRY"`.

    If the call results in an exception, it saves the exception as the task
    result, and sets the task state to `"FAILURE"`.

    Return a function that takes the following arguments:

        :param uuid: The id of the task.
        :param args: List of positional args to pass on to the function.
        :param kwargs: Keyword arguments mapping to pass on to the function.
        :keyword request: Request dict.

    """

    # pylint: disable=too-many-statements

    # If the task doesn't define a custom __call__ method
    # we optimize it away by simply calling the run method directly,
    # saving the extra method call and a line less in the stack trace.
    fun = task if task_has_custom(task, '__call__') else task.run

    loader = loader or app.loader
    deduplicate_successful_tasks = ((app.conf.task_acks_late or task.acks_late)
                                    and app.conf.worker_deduplicate_successful_tasks
                                    and app.backend.persistent)

    hostname = hostname or gethostname()
    inherit_parent_priority = app.conf.task_inherit_parent_priority

    loader_task_init = loader.on_task_init
    loader_cleanup = loader.on_process_cleanup

    task_before_start = None
    task_on_success = None
    task_after_return = None
    if task_has_custom(task, 'before_start'):
        task_before_start = task.before_start
    if task_has_custom(task, 'on_success'):
        task_on_success = task.on_success
    if task_has_custom(task, 'after_return'):
        task_after_return = task.after_return

    pid = os.getpid()

    request_stack = task.request_stack
    push_request = request_stack.push
    pop_request = request_stack.pop
    push_task = _task_stack.push
    pop_task = _task_stack.pop
    _does_info = logger.isEnabledFor(logging.INFO)
    resultrepr_maxsize = task.resultrepr_maxsize

    prerun_receivers = signals.task_prerun.receivers
    postrun_receivers = signals.task_postrun.receivers
    success_receivers = signals.task_success.receivers

    from celery import canvas
    signature = canvas.maybe_signature  # maybe_ does not clone if already

    ctx = _TraceContext(
        name=name, task=task, fun=fun, app=app, Info=Info,
        eager=eager, propagate=propagate, monotonic=monotonic,
        trace_ok_t=trace_ok_t, IGNORE_STATES=IGNORE_STATES,
        signature=signature, hostname=hostname, pid=pid,
        loader_task_init=loader_task_init, loader_cleanup=loader_cleanup,
        task_before_start=task_before_start, task_on_success=task_on_success,
        task_after_return=task_after_return,
        push_request=push_request, pop_request=pop_request,
        push_task=push_task, pop_task=pop_task,
        prerun_receivers=prerun_receivers, postrun_receivers=postrun_receivers,
        success_receivers=success_receivers,
        deduplicate_successful_tasks=deduplicate_successful_tasks,
        successful_requests=successful_requests,
        inherit_parent_priority=inherit_parent_priority,
        resultrepr_maxsize=resultrepr_maxsize, _does_info=_does_info,
    )

    def trace_task(uuid, args, kwargs, request=None):
        return _trace_task(ctx, uuid, args, kwargs, request)

    return trace_task


def trace_task(task, uuid, args, kwargs, request=None, **opts):
    """Trace task execution."""
    request = {} if not request else request
    try:
        if task.__trace__ is None:
            task.__trace__ = build_tracer(task.name, task, **opts)
        return task.__trace__(uuid, args, kwargs, request)
    except Reject:
        raise
    except Exception as exc:
        _signal_internal_error(task, uuid, args, kwargs, request, exc)
        return trace_ok_t(report_internal_error(task, exc), TraceInfo(FAILURE, exc), 0.0, None)


def _signal_internal_error(task, uuid, args, kwargs, request, exc):
    """Send a special `internal_error` signal to the app for outside body errors."""
    tb = None
    einfo = None
    try:
        _, _, tb = sys.exc_info()
        einfo = ExceptionInfo()
        einfo.exception = get_pickleable_exception(einfo.exception)
        einfo.type = get_pickleable_etype(einfo.type)
        signals.task_internal_error.send(
            sender=task,
            task_id=uuid,
            args=args,
            kwargs=kwargs,
            request=request,
            exception=exc,
            traceback=tb,
            einfo=einfo,
        )
    finally:
        # MEMORY LEAK FIX: Clean up local references to prevent memory leaks (Issue #8882)
        # Both 'tb' and 'einfo' can hold references to frame objects and their local variables.
        # Explicitly clearing these prevents reference cycles that block garbage collection.
        if tb is not None:
            del tb
        if einfo is not None:
            # Clear traceback frames to ensure consistent cleanup
            traceback_clear(einfo.exception)
            # Break potential reference cycles by deleting the einfo object
            del einfo


def trace_task_ret(name, uuid, request, body, content_type,
                   content_encoding, loads=loads_message, app=None,
                   **extra_request):
    app = app or current_app._get_current_object()
    embed = None
    if content_type:
        accept = prepare_accept_content(app.conf.accept_content)
        args, kwargs, embed = loads(
            body, content_type, content_encoding, accept=accept,
        )
    else:
        args, kwargs, embed = body
    hostname = gethostname()
    request.update({
        'args': args, 'kwargs': kwargs,
        'hostname': hostname, 'is_eager': False,
    }, **embed or {})
    R, I, T, Rstr = trace_task(app.tasks[name],
                               uuid, args, kwargs, request, app=app)
    return (1, R, T) if I else (0, Rstr, T)


def fast_trace_task(task, uuid, request, body, content_type,
                    content_encoding, loads=loads_message, _loc=None,
                    hostname=None, **_):
    _loc = _localized if not _loc else _loc
    embed = None
    try:
        tasks, accept, hostname = _loc
    except ValueError:
        raise RuntimeError(
            "fast_trace_task: worker task registry is empty "
            "(`_loc` was not populated). This normally means "
            "`setup_worker_optimizations()` never ran in this "
            "process, which happens when the process was spawned "
            "rather than forked (e.g. the default prefork pool on "
            "Windows). Try `--pool=solo` or "
            "`--pool=threads`, or ensure `use_fast_trace_task` is "
            "not enabled for this pool type."
        ) from None
    if content_type:
        args, kwargs, embed = loads(
            body, content_type, content_encoding, accept=accept,
        )
    else:
        args, kwargs, embed = body
    request.update({
        'args': args, 'kwargs': kwargs,
        'hostname': hostname, 'is_eager': False,
    }, **embed or {})
    R, I, T, Rstr = tasks[task].__trace__(
        uuid, args, kwargs, request,
    )
    return (1, R, T) if I else (0, Rstr, T)


def report_internal_error(task, exc):
    _type, _value, _tb = sys.exc_info()
    try:
        _value = task.backend.prepare_exception(exc, 'pickle')
        exc_info = ExceptionInfo((_type, _value, _tb), internal=True)
        warn(RuntimeWarning(
            'Exception raised outside body: {!r}:\n{}'.format(
                exc, exc_info.traceback)))
        return exc_info
    finally:
        del _tb


def setup_worker_optimizations(app, hostname=None):
    """Setup worker related optimizations."""
    hostname = hostname or gethostname()

    # make sure custom Task.__call__ methods that calls super
    # won't mess up the request/task stack.
    _install_stack_protection()

    # all new threads start without a current app, so if an app is not
    # passed on to the thread it will fall back to the "default app",
    # which then could be the wrong app.  So for the worker
    # we set this to always return our app.  This is a hack,
    # and means that only a single app can be used for workers
    # running in the same process.
    app.set_current()
    app.set_default()

    # evaluate all task classes by finalizing the app.
    app.finalize()

    # set fast shortcut to task registry
    _localized[:] = [
        app._tasks,
        prepare_accept_content(app.conf.accept_content),
        hostname,
    ]

    app.use_fast_trace_task = True


def reset_worker_optimizations(app=current_app):
    """Reset previously configured optimizations."""
    try:
        delattr(BaseTask, '_stackprotected')
    except AttributeError:
        pass
    try:
        BaseTask.__call__ = _patched.pop('BaseTask.__call__')
    except KeyError:
        pass
    app.use_fast_trace_task = False


def _install_stack_protection():
    # Patches BaseTask.__call__ in the worker to handle the edge case
    # where people override it and also call super.
    #
    # - The worker optimizes away BaseTask.__call__ and instead
    #   calls task.run directly.
    # - so with the addition of current_task and the request stack
    #   BaseTask.__call__ now pushes to those stacks so that
    #   they work when tasks are called directly.
    #
    # The worker only optimizes away __call__ in the case
    # where it hasn't been overridden, so the request/task stack
    # will blow if a custom task class defines __call__ and also
    # calls super().
    if not getattr(BaseTask, '_stackprotected', False):
        _patched['BaseTask.__call__'] = orig = BaseTask.__call__

        def __protected_call__(self, *args, **kwargs):
            stack = self.request_stack
            req = stack.top
            if req and not req._protected and \
                    len(stack) == 1 and not req.called_directly:
                req._protected = 1
                return self.run(*args, **kwargs)
            return orig(self, *args, **kwargs)
        BaseTask.__call__ = __protected_call__
        BaseTask._stackprotected = True
