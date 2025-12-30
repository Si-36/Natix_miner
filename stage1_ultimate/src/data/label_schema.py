"""
Label Schema - Single Source of Truth for NATIX Classes

CRITICAL: This is the ONLY place where class definitions exist.
All other modules (dataset, split_generator, head, metrics) MUST import from here.

Latest 2025-2026 practices:
- Enum for type safety
- Frozen dataclass for immutability
- Comprehensive validation
- Zero duplicate definitions
"""

from enum import Enum
from dataclasses import dataclass
from typing import ClassVar


class RoadworkClass(Enum):
    """
    NATIX Roadwork (binary) classes (HuggingFace: natix-network-org/roadwork)

    NOTE: This repo is currently standardized on binary classification:
    - 0: no roadwork
    - 1: roadwork

    CRITICAL: This is the canonical definition for Stage-1 binary pipeline.
    """

    NO_ROADWORK = 0
    ROADWORK = 1

    @classmethod
    def from_name(cls, name: str) -> "RoadworkClass":
        """
        Get class from string name

        Args:
            name: Class name (case-insensitive)

        Returns:
            RoadworkClass enum

        Raises:
            ValueError: If name is unknown
        """
        # Normalize name
        name_normalized = name.lower().strip()

        # Try direct mapping
        name_to_class = {
            "no_roadwork": cls.NO_ROADWORK,
            "class_0": cls.NO_ROADWORK,
            "0": cls.NO_ROADWORK,
            "roadwork": cls.ROADWORK,
            "class_1": cls.ROADWORK,
            "1": cls.ROADWORK,
        }

        if name_normalized in name_to_class:
            return name_to_class[name_normalized]

        # CRITICAL: Fail-fast instead of guessing
        raise ValueError(
            f"Unknown class name: '{name}'\n"
            f"Valid names: {list(name_to_class.keys())}\n"
            f"If this is a new class, add it to label_schema.py"
        )

    @classmethod
    def get_display_name(cls, class_id: int) -> str:
        """Get human-readable display name"""
        display_names = {
            0: "No Roadwork",
            1: "Roadwork",
        }
        if class_id not in display_names:
            raise ValueError(f"Invalid class ID: {class_id} (must be 0-1)")
        return display_names[class_id]


@dataclass(frozen=True, slots=True)
class LabelSchema:
    """
    Complete label schema for NATIX dataset

    This is the single source of truth for all class-related constants.

    CRITICAL: All modules must use this instead of hardcoding values.
    """

    NUM_CLASSES: ClassVar[int] = 2
    CLASS_NAMES: ClassVar[list[str]] = [
        "class_0",  # no_roadwork
        "class_1",  # roadwork
    ]

    @classmethod
    def validate_label(cls, label: int) -> bool:
        """
        Validate that label is in valid range

        Args:
            label: Integer label (0-1)

        Returns:
            True if valid

        Raises:
            ValueError: If label is out of range
        """
        if not (0 <= label < cls.NUM_CLASSES):
            raise ValueError(
                f"Label {label} out of range [0, {cls.NUM_CLASSES})\n"
                f"Valid range: 0-{cls.NUM_CLASSES-1}"
            )
        return True

    @classmethod
    def get_class_id(cls, class_name: str) -> int:
        """
        Get class ID from name

        Args:
            class_name: Class name (e.g., "pothole")

        Returns:
            Integer class ID (0-1)

        Raises:
            ValueError: If class name is unknown
        """
        return RoadworkClass.from_name(class_name).value

    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        """
        Get class name from ID

        Args:
            class_id: Integer class ID (0-1)

        Returns:
            Class name string

        Raises:
            ValueError: If class ID is invalid
        """
        cls.validate_label(class_id)
        return cls.CLASS_NAMES[class_id]

    @classmethod
    def get_display_name(cls, class_id: int) -> str:
        """
        Get human-readable display name

        Args:
            class_id: Integer class ID (0-1)

        Returns:
            Display name (e.g., "Pothole")
        """
        return RoadworkClass.get_display_name(class_id)

    @classmethod
    def get_all_classes(cls) -> list[tuple[int, str, str]]:
        """
        Get all classes with ID, name, and display name

        Returns:
            List of (id, name, display_name) tuples
        """
        return [
            (i, cls.get_class_name(i), cls.get_display_name(i))
            for i in range(cls.NUM_CLASSES)
        ]


# Convenience exports
NUM_CLASSES = LabelSchema.NUM_CLASSES
CLASS_NAMES = LabelSchema.CLASS_NAMES


if __name__ == "__main__":
    # Test label schema
    print("=" * 80)
    print("NATIX Label Schema")
    print("=" * 80)

    print(f"\nNumber of classes: {NUM_CLASSES}")

    print("\nAll classes:")
    for class_id, class_name, display_name in LabelSchema.get_all_classes():
        print(f"  {class_id:2d}: {class_name:20s} → {display_name}")

    # Test from_name
    print("\n" + "=" * 80)
    print("Testing class name resolution:")
    print("=" * 80)

    test_names = ["pothole", "Alligator_Crack", "NO_DAMAGE", "invalid_name"]
    for name in test_names:
        try:
            cls = RoadworkClass.from_name(name)
            print(f"✅ '{name}' → {cls.value} ({cls.name})")
        except ValueError as e:
            print(f"❌ '{name}' → ERROR: {e}")

    # Test validation
    print("\n" + "=" * 80)
    print("Testing label validation:")
    print("=" * 80)

    test_labels = [0, 5, 12, 13, -1]
    for label in test_labels:
        try:
            LabelSchema.validate_label(label)
            print(f"✅ Label {label} is valid")
        except ValueError as e:
            print(f"❌ Label {label} is invalid: {e}")

    print("\n" + "=" * 80)
    print("✅ Label schema tests complete!")
    print("=" * 80)
