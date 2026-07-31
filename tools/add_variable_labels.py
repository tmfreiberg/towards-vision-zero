"""Add a short human-readable name to each variable in the codebook.

The codebook records what every variable means, at the length the SAAQ
documentation states it — a sentence or more. That is right for a reference and
wrong for a form label, where a field needs a few words.

The French names here are the SAAQ's own, taken verbatim from the *description
variable* column of the published data dictionary. The English are their
translations, kept to the same length and register. Nothing is invented: this
adds a shorter form of text the codebook already carries.

Run once, from the repository root::

    uv run python tools/add_variable_labels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Variable name to (English, French). French from the published data dictionary.
LABELS: dict[str, tuple[str, str]] = {
    "YEAR": ("Year", "Année"),
    "ID": ("Report number", "Numéro du rapport"),
    "MONTH": ("Month", "Mois"),
    "HOUR": ("Time of day", "Heure"),
    "WKDY_WKND": ("Day of week", "Jour de la semaine"),
    "SEVERITY": ("Severity", "Gravité"),
    "NUM_VICTIMS": ("Victims", "Nombre de victimes"),
    "NUM_VEH": ("Vehicles involved", "Véhicules impliqués"),
    "REGION": ("Administrative region", "Région administrative"),
    "SPD_LIM": ("Speed limit", "Vitesse autorisée"),
    "ACCDN_TYPE": ("Type of collision", "Genre d'accident"),
    "RD_COND": ("Road surface", "État de la surface"),
    "LIGHT": ("Lighting", "Éclairement"),
    "ZONE": ("Surroundings", "Environnement"),
    "PUB_PRIV_RD": ("Road category", "Catégorie de route"),
    "ASPECT": ("Road alignment", "Aspect de la route"),
    "LONG_LOC": ("Location along road", "Localisation longitudinale"),
    "RD_CONFG": ("Road configuration", "Configuration de la route"),
    "RDWX": ("Work zone", "Zone de travaux"),
    "WEATHER": ("Weather", "Conditions météorologiques"),
    "LT_TRK": ("Car or light truck", "Automobile ou camion léger"),
    "HVY_VEH": ("Heavy vehicle", "Véhicule lourd"),
    "MTRCYC": ("Motorcycle or moped", "Motocyclette ou cyclomoteur"),
    "BICYC": ("Bicycle", "Bicyclette"),
    "PED": ("Pedestrian victim", "Victime piéton"),
}


class _Dumper(yaml.SafeDumper):
    """Dumper that indents block sequences under their key."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:  # noqa: ARG002
        super().increase_indent(flow=flow, indentless=False)


def _str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Emit multi-line strings as literal blocks so glosses stay readable."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_presenter)


def main() -> int:
    """Insert the labels and rewrite the codebook. Returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebook",
        type=Path,
        default=Path("src/vision_zero/data/codebook/saaq.yaml"),
    )
    args = parser.parse_args()

    with args.codebook.open(encoding="utf-8") as handle:
        codebook = yaml.safe_load(handle)

    variables = codebook["variables"]
    missing = set(variables) - set(LABELS)
    if missing:
        print(f"No label supplied for: {sorted(missing)}")
        return 1

    for name, entry in variables.items():
        english, french = LABELS[name]
        # Placed after `source_column` so the file reads name, source, label.
        rebuilt = {}
        for key, value in entry.items():
            rebuilt[key] = value
            if key == "source_column":
                rebuilt["label"] = {"en": english, "fr": french}
        variables[name] = rebuilt

    with args.codebook.open("w", encoding="utf-8") as handle:
        yaml.dump(codebook, handle, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=88)

    print(f"Labelled {len(variables)} variables in {args.codebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
