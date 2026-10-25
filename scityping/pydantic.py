"""
Tie-ins for pydantic which explicitely depend on pydantic types.
Provides:

- An scityping-aware "default" callable for `json.dump`, which allows it to
  serialize `Serializable` types.
- Drop-in replacements for `pydantic.BaseModel` and
  `pydantic.dataclasses.dataclass`, which by default uses the scityping JSON
  encoder provided by this package.
"""
from typing import TYPE_CHECKING, Any, Type
from functools import partial
from dataclasses import is_dataclass
from importlib.metadata import version
from typing import get_origin, get_args
from pydantic import (BaseModel as PydanticBaseModel,
                      ValidationError as PydanticValidationError,
                      ConfigDict,
                      model_validator)

def _coerce_int(val, tp):
    if tp is int and isinstance(val, float):
        return int(val)
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (list, tuple) and isinstance(val, (list, tuple)):
        if args:
            sub_t = args[0]
            return type(val)(_coerce_int(item, sub_t) for item in val)
    return val
from pydantic._internal._model_construction import ModelMetaclass as PydanticModelMetaclass
from pydantic.dataclasses import dataclass as pydantic_dataclass
from .base import Serializable, json_like, Dataclass
from .json import scityping_encoder

if TYPE_CHECKING:
    from pydantic.typing import ReprArgs

ValidationError = PydanticValidationError

def remove_overridden_validators(model: PydanticBaseModel) -> PydanticBaseModel:
    return model

def _make_dataclass_core_schema(T: type):
    def get_core_schema(cls, source_type, handler):
        from pydantic_core import core_schema
        schema = handler(source_type)
        return core_schema.no_info_before_validator_function(
            lambda val: Dataclass.validate(val) if json_like(val, Dataclass._registry) else val,
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                Dataclass.deep_reduce,
                info_arg=False,
                return_schema=core_schema.any_schema()
            )
        )
    return get_core_schema

def _register_dataclass_core_schema(T):
    if isinstance(T, type) and is_dataclass(T) and not issubclass(T, Serializable):
        if not hasattr(T, "__get_pydantic_core_schema__"):
            try:
                T.__get_pydantic_core_schema__ = classmethod(_make_dataclass_core_schema(T))
            except (AttributeError, TypeError):
                pass
    for arg in getattr(T, "__args__", ()) or ():
        _register_dataclass_core_schema(arg)
class ModelMetaclass(PydanticModelMetaclass):
    """
    Use this as the metaclass for BaseModel types, to allow them
    1. to find serializers defined in Serializable;
    2. to allow their subclasses to override root validators.
    """
    def __new__(mcs, name, bases, namespace, **kwargs):
        ann = namespace.get("__annotations__", {})
        for field, T in ann.items():
            _register_dataclass_core_schema(T)
        return super().__new__(mcs, name, bases, namespace, **kwargs)

class BaseModel(PydanticBaseModel, metaclass=ModelMetaclass):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    @classmethod
    def _coerce_v1_types(cls, data: Any) -> Any:
        if isinstance(data, dict):
            new_data = dict(data)
            for name, field in cls.model_fields.items():
                if name in new_data and field.annotation is not None:
                    new_data[name] = _coerce_int(new_data[name], field.annotation)
            return new_data
        return data

    def dict(self, *args, **kwargs):
        return self.model_dump(*args, **kwargs)

    def json(self, *args, **kwargs):
        return self.model_dump_json(*args, **kwargs)

    @classmethod
    def parse_raw(cls, b, *args, **kwargs):
        return cls.model_validate_json(b, *args, **kwargs)

    @classmethod
    def parse_obj(cls, obj, *args, **kwargs):
        return cls.model_validate(obj, *args, **kwargs)

class GenericModel(BaseModel):
    pass

def _get_dataclass_validator(T: type):
    def _dataclass_type_match(value):
        if not isinstance(value, T):
            raise TypeError(f"Value is not of type `{T.__qualname__}`")
        return value
    def _get_validators():
        yield _dataclass_deserializer
        yield _dataclass_type_match
    return _get_validators

def _dataclass_deserializer(value):
    if json_like(value, Dataclass._registry):
        return Dataclass.validate(value)
    else:
        return value

def _validate_serializable_dataclass(cls: Type["DataclassT"], v: Any) -> "DataclassT":
    if json_like(v, cls._registry):
        return cls.validate(v)
    else:
        return cls(**v) if isinstance(v, dict) else v

def _validate_plain_dataclass(cls: Type["DataclassT"], v: Any) -> "DataclassT":
    if json_like(v, Dataclass._registry):
        return Dataclass.validate(v)
    else:
        return cls(**v) if isinstance(v, dict) else v

def dataclass(_cls=None, **kwargs):
    ## Secondary code path: decorator with arguments ##
    if _cls is None:
        return partial(dataclass, **kwargs)

    ## Primary code path: decorator without arguments ##
    dclass = pydantic_dataclass(_cls, **kwargs)
    _register_dataclass_core_schema(dclass)
    return dclass
