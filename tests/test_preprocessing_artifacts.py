import unittest

import cv2
import numpy as np

from preprocessing.signature_preprocessing import (
    crop_to_ink,
    preprocess_signature,
    remove_border_artifacts,
)


def _page() -> np.ndarray:
    page = np.full((300, 1000, 3), 255, np.uint8)
    points = np.array([[50, 150], [120, 100], [180, 160], [240, 105], [300, 150]], np.int32)
    cv2.polylines(page, [points], False, (0, 0, 0), thickness=7)  # the "signature"
    return page


class BorderArtifactTests(unittest.TestCase):
    def test_edge_line_is_removed_and_signature_kept(self) -> None:
        gray = _page()[..., 0]
        gray[0:3, 400:990] = 0
        cleaned = remove_border_artifacts(gray)
        self.assertEqual(int((cleaned[0:3] < 200).sum()), 0)
        self.assertEqual(int((cleaned[40:] < 200).sum()), int((gray[40:] < 200).sum()))

    def test_signature_touching_nothing_is_untouched(self) -> None:
        gray = _page()[..., 0]
        self.assertTrue(np.array_equal(remove_border_artifacts(gray), gray))

    def test_edge_line_does_not_shrink_the_signature_in_output(self) -> None:
        clean = _page()
        noisy = _page()
        noisy[0:3, 400:990] = 0
        self.assertTrue(
            np.array_equal(
                preprocess_signature(clean, binarize=True),
                preprocess_signature(noisy, binarize=True),
            )
        )

    def test_isolated_speck_does_not_stretch_crop(self) -> None:
        gray = _page()[..., 0]
        gray[10:13, 900:903] = 0
        cropped = crop_to_ink(gray)
        self.assertLess(cropped.shape[1], 400)

    def test_far_stray_mark_does_not_stretch_crop_but_near_dot_does_not_vanish(self) -> None:
        gray = _page()[..., 0]
        gray[20:26, 950:956] = 0  # dust far from the writing, bigger than the speck limit
        cropped = crop_to_ink(gray)
        self.assertLess(cropped.shape[1], 400)

        near = _page()[..., 0]
        near[60:66, 330:336] = 0  # a dot just beside the writing
        cropped = crop_to_ink(near)
        self.assertGreaterEqual(cropped.shape[1], 300)

    def test_edge_line_bridged_to_the_writing_is_removed(self) -> None:
        gray = _page()[..., 0]
        gray[0:8, 100:900] = 0
        gray[0:100, 60:64] = 0  # bridge from the line down to the signature
        cleaned = remove_border_artifacts(gray)
        self.assertEqual(int((cleaned[0:8, 120:900] < 200).sum()), 0)
        self.assertGreater(int((cleaned[100:170] < 200).sum()), 0)

    def test_short_stroke_touching_the_edge_is_kept(self) -> None:
        gray = np.full((300, 1000), 255, np.uint8)
        gray[100:180, 0:6] = 0  # short vertical stem at the left edge
        gray[150:160, 0:40] = 0
        cleaned = remove_border_artifacts(gray)
        self.assertTrue(np.array_equal(cleaned, gray))


if __name__ == "__main__":
    unittest.main()
