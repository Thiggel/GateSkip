import argparse
import re
from pydantic import BaseModel
from typing import (
    Type,
    Any,
    Optional,
    Union,
    get_origin,
    get_args,
    Literal,
    List,
    Annotated,
)
from enum import Enum


class CLIManager:
    """
    Enhanced wrapper around argparse to dynamically generate CLI options from Pydantic models
    with support for custom types and field constraints.
    """

    def __init__(self, *config_classes: Type[BaseModel]):
        self.config_classes = config_classes
        self.configs: dict[str, BaseModel] = {}

    def _parse_list(self, value: str, item_type: Type) -> list:
        """Parse comma-separated values into a list of the specified type"""
        if not value:
            return []
        try:
            items = value.replace(",", " ").split()
            return [item_type(item) for item in items]
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid list item: {e}")

    def _get_literal_values(self, args):
        """Extract all literal values from Union args"""
        literal_values = []
        for arg in args:
            if get_origin(arg) is Literal:
                literal_values.extend(arg.__args__)
            elif hasattr(arg, "__args__") and isinstance(arg.__args__[0], str):
                # Handle cases where Literal is nested in another type
                literal_values.extend(arg.__args__)
        return literal_values

    def _get_type_parser(self, field_type: Any, field: Any) -> tuple[Any, dict]:
        """
        Determine the appropriate parser and argparse kwargs for a given field type
        """
        origin = get_origin(field_type)
        args = get_args(field_type)
        kwargs = {"help": field.description or ""}

        # Handle bool type
        if field_type is bool:
            if field.default is True:
                # Return None to skip adding the positive flag (since it's default True)
                return None, None
            else:
                kwargs["action"] = "store_true"
                return (None, kwargs)  # Handle Annotated types

        if origin is Annotated:
            base_type, *custom_metadata = args
            for metadata in custom_metadata:
                if hasattr(metadata, "parse"):
                    return (metadata.parse, kwargs)
            return self._get_type_parser(base_type, field)

        # Handle Union types
        if origin is Union:
            if type(None) in args:
                args = tuple(t for t in args if t is not type(None))
                kwargs["required"] = False

            # Get all literal values first
            literal_values = self._get_literal_values(args)
            if literal_values:

                def literal_parser(value):
                    if value in literal_values:
                        return value
                    # Try other types if not a literal
                    for arg in args:
                        if arg is not type(None) and not isinstance(arg, type(Literal)):
                            try:
                                if get_origin(arg) is list:
                                    return self._parse_list(value, get_args(arg)[0])
                                return arg(value)
                            except (ValueError, TypeError):
                                continue
                    raise argparse.ArgumentTypeError(
                        f"Value must be one of {literal_values} or valid {[arg for arg in args if arg is not type(None) and not isinstance(arg, type(Literal))]}"
                    )

                kwargs["help"] = f"{kwargs['help']} (literals: {literal_values})"
                return (literal_parser, kwargs)

            # Handle other union types
            parsers = []
            for arg in args:
                if get_origin(arg) is list:
                    parsers.append(lambda x: self._parse_list(x, get_args(arg)[0]))
                elif isinstance(arg, type) and issubclass(arg, Enum):
                    parsers.append(arg)
                else:
                    parsers.append(arg)

            def union_parser(value):
                for parser in parsers:
                    try:
                        return parser(value)
                    except (ValueError, argparse.ArgumentTypeError):
                        continue
                raise argparse.ArgumentTypeError(
                    f"Could not parse '{value}' as any of {args}"
                )

            return (union_parser, kwargs)

        # Handle List types
        if origin is list or origin is List:
            item_type = args[0]
            kwargs["nargs"] = "+"
            if item_type is int:
                return (int, kwargs)
            return (item_type, kwargs)

        # Handle Enum types
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            kwargs["choices"] = [e.value for e in field_type]
            return (str, kwargs)

        # Handle pattern-constrained strings
        if hasattr(field, "pattern") and field.pattern:

            def pattern_validator(value):
                if not re.match(field.pattern, value):
                    raise argparse.ArgumentTypeError(
                        f"Value '{value}' does not match pattern '{field.pattern}'"
                    )
                return value

            return (pattern_validator, kwargs)

        return (field_type, kwargs)

    def _add_bool_argument(self, parser, name: str, field: Any):
        """Add a boolean argument with appropriate --flag/--no-flag options"""
        cli_name = f"--{name.replace('_', '-')}"
        dest = name.replace("-", "_")

        if field.default is True:
            # For default=True, only add the --no- variant
            parser.add_argument(
                f"--no-{name.replace('_', '-')}",
                dest=dest,
                action="store_false",
                help=f"{field.description or ''} (default: True)",
            )
        else:
            # For default=False, add the regular flag
            parser.add_argument(
                cli_name,
                dest=dest,
                action="store_true",
                help=field.description,
                default=False,
            )

    def parse_args(self):
        """Parse command-line arguments and populate configuration models."""
        parser = argparse.ArgumentParser(description="Configuration CLI")

        for config_class in self.config_classes:
            for name, field in config_class.model_fields.items():
                if field.annotation is bool:
                    self._add_bool_argument(parser, name, field)
                else:
                    cli_name = f"--{name.replace('_', '-')}"
                    type_parser, kwargs = self._get_type_parser(field.annotation, field)
                    if type_parser is not None and kwargs is not None:
                        if field.default is not None:
                            kwargs["default"] = field.default
                        if type_parser is not None:
                            kwargs["type"] = type_parser
                        parser.add_argument(cli_name, **kwargs)
        args = parser.parse_args()

        # Populate configs with parsed arguments
        for config_class in self.config_classes:
            config_args = {
                name: getattr(args, name.replace("-", "_"))
                for name in config_class.model_fields.keys()
            }
            self.configs[config_class.__name__] = config_class(**config_args)

    def get_config(self, config_name: str) -> Optional[BaseModel]:
        """Retrieve a populated config by class name."""
        return self.configs.get(config_name)

    def get_all_configs(self) -> dict[str, BaseModel]:
        """Retrieve all populated config objects."""
        return self.configs
