import unittest

from utils.variant_selection import resolve_selection


class VariantSelectionTest(unittest.TestCase):
    def test_single_and_full_all_keep_readable_tags(self):
        available = ["open", "umaze", "medium"]

        single = resolve_selection(
            mode="single",
            variants=["open"],
            available_variants=available,
            field_name="variants",
        )
        self.assertEqual(single.selection_tag, "open")
        self.assertEqual(single.full_selection_tag, "open")

        all_variants = resolve_selection(
            mode="all",
            variants=[],
            available_variants=available,
            field_name="variants",
        )
        self.assertEqual(all_variants.selection_tag, "all")
        self.assertEqual(all_variants.full_selection_tag, "all")

    def test_all_subset_uses_compact_order_invariant_tag(self):
        available = ["open", "umaze", "medium", "large"]

        first = resolve_selection(
            mode="all",
            variants=["medium", "open"],
            available_variants=available,
            field_name="variants",
        )
        second = resolve_selection(
            mode="all",
            variants=["open", "medium"],
            available_variants=available,
            field_name="variants",
        )

        self.assertRegex(first.selection_tag, r"^all-2v-[0-9a-f]{12}$")
        self.assertEqual(first.selection_tag, second.selection_tag)
        self.assertEqual(first.full_selection_tag, "all-medium+open")
        self.assertNotIn("medium+open", first.selection_tag)

    def test_except_uses_compact_order_invariant_tag(self):
        available = ["open", "umaze", "medium", "large"]

        first = resolve_selection(
            mode="except",
            variants=["large", "medium"],
            available_variants=available,
            field_name="variants",
        )
        second = resolve_selection(
            mode="except",
            variants=["medium", "large"],
            available_variants=available,
            field_name="variants",
        )

        self.assertRegex(first.selection_tag, r"^except-2x-[0-9a-f]{12}$")
        self.assertEqual(first.selection_tag, second.selection_tag)
        self.assertEqual(first.full_selection_tag, "except-large+medium")
        self.assertEqual(first.selected_variants, ["open", "umaze"])
        self.assertNotIn("large+medium", first.selection_tag)


if __name__ == "__main__":
    unittest.main()
