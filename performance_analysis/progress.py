"""Show comparison progress while hiding nested simulation bars."""

from contextlib import contextmanager
from unittest.mock import patch

from tqdm.auto import tqdm
from tqdm.std import tqdm as tqdm_base


@contextmanager
def comparison_progress(description, total):
    original_init = tqdm_base.__init__

    def disabled_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original_init(self, *args, **kwargs)

    with (
        tqdm(
            desc=description,
            total=total,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
            dynamic_ncols=True,
        ) as progress,
        patch.object(tqdm_base, "__init__", disabled_init),
    ):
        yield progress
