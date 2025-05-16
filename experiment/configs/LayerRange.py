from typing import Union
import argparse


class LayerRange:
    """Custom type for layer ranges that can be a single number or range"""

    @staticmethod
    def parse(value: str) -> Union[int, tuple[int, int]]:
        if ":" in value:
            try:
                start, end = map(int, value.split(":"))
                if start >= end:
                    raise ValueError("Start must be less than end")
                return (start, end)
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"Invalid range format: {e}")
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer: {value}")
