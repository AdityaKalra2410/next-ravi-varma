# Dataset

**Source:** Wikimedia Commons, "Category:Paintings by Raja Ravi Varma" and its subcategories.
**License:** 89 images public domain (artist died 1906), 5 under Creative Commons (CC BY / CC BY-SA, incl. Wellcome Collection). Source URL and license of every image are in `metadata.csv`.

**Images:** `data/images/` (94 files).

## How it was built
1. `scripts/scrape_commons.py` found 319 files and kept 135 (JPG/PNG, shorter side at least 768 px).
2. Removed 41 images:
   - 37 duplicate scans of the same painting (found with `scripts/dedupe.py`, then checked by hand)
   - 4 non-paintings (gallery photo, sculpture photo, magazine page, a later studio work)
3. Labelled each remaining image as an oil painting or a printed copy (oleograph), using Commons titles and visual cues (flat colours and printed text vs. blended brushwork and canvas texture). 5 uncertain cases are flagged as low confidence.

## Final numbers
| | Count |
|---|---|
| Files found | 319 |
| Downloaded (size/format filter) | 135 |
| After cleaning | 94 |
| Oil paintings (used for training) | 62 |
| Oleographs (kept aside for comparison) | 32 |

## metadata.csv columns
`filename, commons_title, title, date, license, orig_width, orig_height, source_url, type (oil/oleograph), type_confidence, keep (y/n), drop_reason, split`
