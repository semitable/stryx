from __future__ import annotations
from dataclasses import dataclass, field
import stryx

class TestFromDataclass:
    """Test stryx.from_dataclass() conversion."""

    def test_simple_dataclass(self):
        """Simple dataclass converts to working Pydantic model."""
        @dataclass
        class SimpleConfig:
            lr: float = 1e-4
            name: str = "simple"

        Model = stryx.from_dataclass(SimpleConfig)

        # Can instantiate with defaults
        instance = Model()
        assert instance.lr == 1e-4
        assert instance.name == "simple"

        # Can override
        instance = Model(lr=0.001)
        assert instance.lr == 0.001

    def test_nested_dataclass(self):
        """Nested dataclasses convert recursively."""
        @dataclass
        class Inner:
            x: int = 1

        @dataclass
        class Outer:
            inner: Inner = field(default_factory=Inner)

        Model = stryx.from_dataclass(Outer)
        instance = Model()

        assert instance.inner.x == 1

    def test_consistent_types(self):
        """Same dataclass converts to same Pydantic model."""
        @dataclass
        class MyConfig:
            value: int = 42

        Model1 = stryx.from_dataclass(MyConfig)
        Model2 = stryx.from_dataclass(MyConfig)

        # Should be the exact same class (cached)
        assert Model1 is Model2

    def test_shared_nested_types(self):
        """Nested dataclasses shared across parents get same type."""
        @dataclass
        class Shared:
            value: int = 1

        @dataclass
        class Parent1:
            shared: Shared = field(default_factory=Shared)

        @dataclass
        class Parent2:
            shared: Shared = field(default_factory=Shared)

        Model1 = stryx.from_dataclass(Parent1)
        Model2 = stryx.from_dataclass(Parent2)

        # The Shared type should be the same in both
        instance1 = Model1()
        instance2 = Model2()

        # Both should accept each other's inner type
        assert type(instance1.shared) is type(instance2.shared)
