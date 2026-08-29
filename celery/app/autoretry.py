"""Tasks auto-retry functionality."""
from vine.utils import wraps

from celery.exceptions import Ignore, Retry
from celery.utils.time import get_exponential_backoff_interval


def _resolve_autoretry_option(task, options, name, default, cast=None):
    """Read an autoretry option from ``options`` then the task attribute."""
    value = options.get(name, getattr(task, name, default))
    return cast(value) if cast is not None else value


def _build_retry_kwargs(task, retry_kwargs, retry_backoff,
                        retry_backoff_max, retry_jitter):
    """Build the per-attempt ``retry`` kwargs for a failing task."""
    retry_kwargs_for_attempt = retry_kwargs.copy()
    if retry_backoff:
        retry_kwargs_for_attempt['countdown'] = \
            get_exponential_backoff_interval(
                factor=int(max(1.0, retry_backoff)),
                retries=task.request.retries,
                maximum=retry_backoff_max,
                full_jitter=retry_jitter)
    if hasattr(task, 'override_max_retries'):
        retry_kwargs_for_attempt['max_retries'] = getattr(
            task, 'override_max_retries', task.max_retries)
    return retry_kwargs_for_attempt


def add_autoretry_behaviour(task, **options):
    """Wrap task's `run` method with auto-retry functionality."""
    autoretry_for = tuple(
        _resolve_autoretry_option(task, options, 'autoretry_for', ()))
    dont_autoretry_for = tuple(
        _resolve_autoretry_option(task, options, 'dont_autoretry_for', ()))
    retry_kwargs = _resolve_autoretry_option(
        task, options, 'retry_kwargs', {})
    retry_backoff = _resolve_autoretry_option(
        task, options, 'retry_backoff', False, cast=float)
    retry_backoff_max = _resolve_autoretry_option(
        task, options, 'retry_backoff_max', 600, cast=int)
    retry_jitter = _resolve_autoretry_option(
        task, options, 'retry_jitter', True)

    if not autoretry_for or hasattr(task, '_orig_run'):
        return

    # Exceptions that must always propagate instead of triggering a retry:
    # Ignore/Retry are handled elsewhere, and dont_autoretry_for exceptions
    # are explicitly excluded from auto-retry.
    never_autoretry_for = (Ignore, Retry) + dont_autoretry_for

    @wraps(task.run)
    def run(*args, **kwargs):
        try:
            return task._orig_run(*args, **kwargs)
        except never_autoretry_for:
            raise
        except autoretry_for as exc:
            retry_kwargs_for_attempt = _build_retry_kwargs(
                task, retry_kwargs, retry_backoff,
                retry_backoff_max, retry_jitter)
            ret = task.retry(exc=exc, **retry_kwargs_for_attempt)
            # Stop propagation
            if hasattr(task, 'override_max_retries'):
                delattr(task, 'override_max_retries')
            raise ret

    task._orig_run, task.run = task.run, run
