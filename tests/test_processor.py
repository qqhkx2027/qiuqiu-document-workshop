import unittest

import cv2
import numpy as np
from pypdf import PdfReader

from app import CARD_HEIGHT_PX, CARD_WIDTH_PX, crop_card, create_pdf


def synthetic_card(color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.full((900, 1400, 3), 242, dtype=np.uint8)
    points = np.array([[240, 180], [1135, 125], [1185, 660], [200, 720]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, points, color)
    cv2.polylines(canvas, [points], True, (20, 20, 20), 8)
    return canvas


class ProcessorTests(unittest.TestCase):
    def test_auto_crop_returns_standard_pixel_ratio(self):
        result = crop_card(synthetic_card((210, 165, 120)))
        self.assertEqual(result.shape[:2], (CARD_HEIGHT_PX, CARD_WIDTH_PX))
        self.assertAlmostEqual(result.shape[1] / result.shape[0], 85.6 / 54.0, places=2)

    def test_pdf_page_is_a4_and_contains_two_cards(self):
        output = create_pdf(crop_card(synthetic_card((210, 165, 120))), crop_card(synthetic_card((120, 165, 210))))
        reader = PdfReader(str(output))
        self.assertEqual(len(reader.pages), 1)
        page = reader.pages[0]
        self.assertAlmostEqual(float(page.mediabox.width), 595.2756, places=1)
        self.assertAlmostEqual(float(page.mediabox.height), 841.8898, places=1)
        self.assertEqual(len(page.images), 2)


if __name__ == "__main__":
    unittest.main()
