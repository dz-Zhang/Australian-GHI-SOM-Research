"""Load the complete GHI archive, regrid it to Australian Albers equal area,
and persist the result for reproducible analyses."""

from pathlib import Path

from analysis.dataloader import DataLoader

SOURCE_DIR = Path("aus_ghi")
OUTPUT_DIR = Path("aus_ghi_transformed")
OUTPUT_FILE = OUTPUT_DIR / "ghi_20200101_20241231_equal_area.nc"


def main() -> Path:
    data = DataLoader.from_file(SOURCE_DIR, regrid_equal_area=True)
    output = data.write(OUTPUT_FILE)
    print(f"Wrote {output} ({dict(data.data.sizes)})")
    return output


if __name__ == "__main__":
    main()
