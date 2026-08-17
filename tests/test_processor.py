import unittest
import io

import cv2
import numpy as np
from PIL import Image
from pypdf import PdfReader

from app import CARD_HEIGHT_PX, CARD_WIDTH_PX, crop_card, create_pdf


def synthetic_card(color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.full((900, 1400, 3), 242, dtype=np.uint8)
    points = np.array([[240, 180], [1135, 125], [1185, 660], [200, 720]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, points, color)
    cv2.polylines(canvas, [points], True, (20, 20, 20), 8)
    return canvas


def desk_photo() -> np.ndarray:
    """A light, rounded card on a dark textured desk, like a phone photo."""
    canvas = np.zeros((820, 1180, 3), dtype=np.uint8)
    for y in range(canvas.shape[0]):
        canvas[y, :, :] = (45 + (y % 17), 58 + (y % 13), 38 + (y % 11))
    points = np.array([[170, 140], [860, 105], [900, 650], [145, 680]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, points, (225, 230, 224))
    cv2.polylines(canvas, [points], True, (190, 198, 193), 8)
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

    def test_bright_card_on_dark_desk_is_not_treated_as_full_photo(self):
        result = crop_card(desk_photo())
        self.assertEqual(result.shape[:2], (CARD_HEIGHT_PX, CARD_WIDTH_PX))
        # The crop should be mostly card-colored, not mostly dark desk.
        self.assertGreater(float(result.mean()), 150.0)

    def test_exif_orientation_is_transposed_before_cv(self):
        from app import _read_upload

        source = np.zeros((120, 200, 3), dtype=np.uint8)
        source[:, :100] = (255, 0, 0)
        source[:, 100:] = (0, 255, 0)
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        exif = image.getexif()
        exif[274] = 6  # rotate 90° clockwise when displayed
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif.tobytes())
        stream.seek(0)

        class Upload:
            def read(self):
                return stream.read()

        loaded = _read_upload(Upload())
        self.assertEqual(loaded.shape[:2], (200, 120))


if __name__ == "__main__":
    unittest.main()
