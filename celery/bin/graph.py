"""The ``celery graph`` command."""
import sys
from collections import namedtuple
from operator import itemgetter

import click

from celery.bin.base import CeleryCommand, handle_preload_options, handle_remote_command_error
from celery.utils.graph import DependencyGraph, GraphFormatter

# Bundles the endpoints and node classes needed while drawing the workers graph.
_GraphContext = namedtuple('_GraphContext', ('broker', 'backend',
                                             'worker_cls', 'thread_cls'))


def _resolve_workers_and_threads(app, args):
    """Return the list of worker names and their thread counts.

    Prefer values passed explicitly via ``args``; otherwise inspect the
    running workers for their pool concurrency.
    """
    try:
        return args['nodes'], (args.get('threads') or [])
    except KeyError:
        pass
    try:
        replies = app.control.inspect().stats() or {}
    except Exception as exc:
        handle_remote_command_error('graph workers', exc)
    worker_names, thread_counts = [], []
    for worker, reply in replies.items():
        worker_names.append(worker)
        thread_counts.append(reply['pool']['max-concurrency'])
    return worker_names, thread_counts


def _resolve_broker_uri(app, args):
    try:
        return args.get('broker', app.connection_for_read().as_uri())
    except Exception as exc:
        handle_remote_command_error('graph workers', exc)


def _add_worker_arcs(deps, workers, threads_for, gctx):
    """Add each worker (and its threads) to the dependency graph."""
    for i, worker in enumerate(workers):
        worker = gctx.worker_cls(worker, pos=i)
        deps.add_arc(worker)
        deps.add_edge(worker, gctx.broker)
        if gctx.backend:
            deps.add_edge(worker, gctx.backend)
        for thread in threads_for.get(worker._label) or ():
            thread = gctx.thread_cls(thread)
            deps.add_arc(thread)
            deps.add_edge(thread, worker)


@click.group()
@click.pass_context
@handle_preload_options
def graph(ctx):
    """The ``celery graph`` command."""


@graph.command(cls=CeleryCommand, context_settings={'allow_extra_args': True})
@click.pass_context
def bootsteps(ctx):
    """Display bootsteps graph."""
    worker = ctx.obj.app.WorkController()
    include = {arg.lower() for arg in ctx.args or ['worker', 'consumer']}
    if 'worker' in include:
        worker_graph = worker.blueprint.graph
        if 'consumer' in include:
            worker.blueprint.connect_with(worker.consumer.blueprint)
    else:
        worker_graph = worker.consumer.blueprint.graph
    worker_graph.to_dot(sys.stdout)


@graph.command(cls=CeleryCommand, context_settings={'allow_extra_args': True})
@click.pass_context
def workers(ctx):
    """Display workers graph."""
    def simplearg(arg):
        return maybe_list(itemgetter(0, 2)(arg.partition(':')))

    def maybe_list(l, sep=','):
        return l[0], l[1].split(sep) if sep in l[1] else l[1]

    args = dict(simplearg(arg) for arg in ctx.args)
    generic = 'generic' in args

    def generic_label(node):
        return '{} ({}://)'.format(type(node).__name__,
                                   node._label.split('://')[0])

    class Node:
        force_label = None
        scheme = {}

        def __init__(self, label, pos=None):
            self._label = label
            self.pos = pos

        def label(self):
            return self._label

        def __str__(self):
            return self.label()

    class Thread(Node):
        scheme = {
            'fillcolor': 'lightcyan4',
            'fontcolor': 'yellow',
            'shape': 'oval',
            'fontsize': 10,
            'width': 0.3,
            'color': 'black',
        }

        def __init__(self, label, **kwargs):
            self.real_label = label
            super().__init__(
                label=f'thr-{next(tids)}',
                pos=0,
            )

    class Formatter(GraphFormatter):

        def label(self, obj):
            return obj and obj.label()

        def node(self, obj):
            scheme = dict(obj.scheme) if obj.pos else obj.scheme
            if isinstance(obj, Thread):
                scheme['label'] = obj.real_label
            return self.draw_node(
                obj, dict(self.node_scheme, **scheme),
            )

        def terminal_node(self, obj):
            return self.draw_node(
                obj, dict(self.term_scheme, **obj.scheme),
            )

        def edge(self, a, b, **attrs):
            if isinstance(a, Thread):
                attrs.update(arrowhead='none', arrowtail='tee')
            return self.draw_edge(a, b, self.edge_scheme, attrs)

    def subscript(n):
        S = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
             '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
        return ''.join([S[i] for i in str(n)])

    class Worker(Node):
        pass

    class Backend(Node):
        scheme = {
            'shape': 'folder',
            'width': 2,
            'height': 1,
            'color': 'black',
            'fillcolor': 'peachpuff3',
        }

        def label(self):
            return generic_label(self) if generic else self._label

    class Broker(Node):
        scheme = {
            'shape': 'circle',
            'fillcolor': 'cadetblue3',
            'color': 'cadetblue4',
            'height': 1,
        }

        def label(self):
            return generic_label(self) if generic else self._label

    from itertools import count
    tids = count(1)
    Wmax = int(args.get('wmax', 4) or 0)
    Tmax = int(args.get('tmax', 3) or 0)

    def maybe_abbr(l, name, max=Wmax):
        size = len(l)
        abbr = max and size > max
        if 'enumerate' in args:
            l = [f'{name}{subscript(i + 1)}'
                 for i, obj in enumerate(l)]
        if abbr:
            l = l[0:max - 1] + [l[size - 1]]
            l[max - 2] = '{}⎨…{}⎬'.format(
                name[0], subscript(size - (max - 1)))
        return l

    def build_threads_for(workers, threads, worker_count):
        """Map each abbreviated worker label to its abbreviated thread list.

        ``worker_count`` is the number of workers *before* abbreviation, so the
        thread list is trimmed in step with how the worker list was reduced.
        """
        if Wmax and worker_count > Wmax:
            threads = threads[0:3] + [threads[-1]]
        threads_for = {}
        for i, thread_count in enumerate(threads):
            threads_for[workers[i]] = maybe_abbr(
                list(range(int(thread_count))), 'P', Tmax,
            )
        return threads_for

    app = ctx.obj.app
    workers, threads = _resolve_workers_and_threads(app, args)
    backend = args.get('backend', app.conf.result_backend)
    worker_count = len(workers)
    workers = maybe_abbr(workers, 'Worker')
    threads_for = build_threads_for(workers, threads, worker_count)

    broker = Broker(_resolve_broker_uri(app, args))
    backend = Backend(backend) if backend else None
    deps = DependencyGraph(formatter=Formatter())
    deps.add_arc(broker)
    if backend:
        deps.add_arc(backend)
    gctx = _GraphContext(broker=broker, backend=backend,
                         worker_cls=Worker, thread_cls=Thread)
    _add_worker_arcs(deps, workers, threads_for, gctx)

    deps.to_dot(sys.stdout)
