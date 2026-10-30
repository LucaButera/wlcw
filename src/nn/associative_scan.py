# From https://github.com/i404788/s5-pytorch/blob/master/s5/jax_compat.py
from functools import partial
from typing import Callable, overload, TypeVar, Iterable, Any

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@overload
def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> list[T]: ...


@overload
def safe_map(
    f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]
) -> list[T]: ...


@overload
def safe_map(
    f: Callable[[T1, T2, T3], T],
    __arg1: Iterable[T1],
    __arg2: Iterable[T2],
    __arg3: Iterable[T3],
) -> list[T]: ...


@overload
def safe_map(
    f: Callable[..., T],
    __arg1: Iterable[Any],
    __arg2: Iterable[Any],
    __arg3: Iterable[Any],
    __arg4: Iterable[Any],
    *args,
) -> list[T]: ...


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


def combine(tree, operator, a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat


def _interleave(a, b, axis: int):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    b_trunc = a.shape[axis] == b.shape[axis] + 1
    if b_trunc:
        pad = [0, 0] * b.ndim
        # +1=always end of dim, pad-order is reversed so start is at end
        pad[(b.ndim - axis - 1) * 2 + 1] = 1
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        interleaved = torch.ops.aten.slice(
            interleaved, axis, 0, b.shape[axis] + a.shape[axis] - 1
        )
    return interleaved


def _scan(tree, operator, elems, axis: int):
    """Perform scan on `elems`."""
    num_elems = elems[0].shape[axis]

    if num_elems < 2:
        return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(
        tree,
        operator,
        [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
        [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems],
    )

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(tree, operator, reduced_elems, axis)

    if num_elems % 2 == 0:
        even_elems = combine(
            tree,
            operator,
            [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )
    else:
        even_elems = combine(
            tree,
            operator,
            odd_elems,
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
        torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
        if result.shape.numel() > 0 and elem.shape[axis] > 0
        else result
        if result.shape.numel() > 0
        # Jax allows/ignores concat with 0-dim, Pytorch does not
        else torch.ops.aten.slice(elem, axis, 0, 1)
        for (elem, result) in zip(elems, even_elems)
    ]

    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))


def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool = False):
    elems_flat, tree = tree_flatten(elems)
    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    assert (
        axis >= 0 or axis < elems_flat[0].ndim
    ), "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format([elem.shape for elem in elems_flat])
        )

    scans = _scan(tree, operator, elems_flat, axis)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(scans, tree)


@torch.jit.script
def linear_recurrence_op(q_i: tuple[Tensor, Tensor], q_j: tuple[Tensor, Tensor]):
    """Binary operator for parallel scan of linear recurrence.
    Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing a_i and u_i at position i       (P, ), (P, )
        q_j: tuple containing a_j and u_j at position j       (P, ), (P, )
    Returns:
        new element ( a_out, u_out )
    """
    a_i, u_i = q_i
    a_j, u_j = q_j
    # return a_j * a_i, a_j * u_i + u_j
    return a_j * a_i, torch.addcmul(u_j, a_j, u_i)
