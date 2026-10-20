"""What the studio can make, and what each generator needs.

The studio is a *view* over the command line, not a second implementation of it.
Every entry here names a ``cli.py`` subcommand and a list of fields; the server turns
the fields into argv, runs the command in a fresh directory and shows the files it
wrote.  That is deliberate: the CLI is the interface a script and CI use, and a
studio that called the generator functions directly would be a second definition of
what an option means.  A form field is a flag, and the command is printed beside the
picture so a result can leave the browser and become a script.

The catalog imports the same small tables the CLI's parsers use -- cycles, facings,
portrait styles, text faces -- so a new choice appears here when it appears there.
Nothing here writes a file and nothing here runs a command; the server does both.
"""
from __future__ import annotations

from gen.figures import figure
from gen.portraits import demography as portrait_demography
from gen.portraits import dna as portrait_dna
from gen.portraits import style as portrait_style
from gen.text import pen as text_pen
from gen.ui import style as ui_style

#: The cloth's hangings, as ``cli.py`` spells them.  "bar" is a crossbar; the
#: command line has always used the short name.
HANGINGS = ("pole", "bar", "staff")

#: Flag silhouettes, from the parser that validates them.
SHAPES = ("rectangle", "swallowtail", "pennant", "guidon")


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _int(name: str, label: str, default: int, lo: int, hi: int, help: str = "") -> dict:
    return {"name": name, "label": label, "type": "int", "default": int(default),
            "min": int(lo), "max": int(hi), "help": help, "flag": _flag(name)}


def _float(name: str, label: str, default: float, lo: float, hi: float,
           step: float = 0.01, help: str = "") -> dict:
    return {"name": name, "label": label, "type": "float", "default": float(default),
            "min": float(lo), "max": float(hi), "step": float(step), "help": help,
            "flag": _flag(name)}


def _choice(name: str, label: str, default: str, choices, help: str = "",
            optional: bool = False) -> dict:
    out = {"name": name, "label": label, "type": "choice",
           "default": default, "choices": list(choices), "help": help,
           "flag": _flag(name)}
    if optional:
        out["optional"] = True
    return out


def _bool(name: str, label: str, flag: str, default: bool = False,
          help: str = "") -> dict:
    return {"name": name, "label": label, "type": "bool", "default": bool(default),
            "flag": flag, "help": help}


def _text(name: str, label: str, default: str = "", help: str = "",
          optional: bool = False, repeat: bool = False) -> dict:
    out = {"name": name, "label": label, "type": "text", "default": default,
           "help": help, "flag": _flag(name)}
    if optional:
        out["optional"] = True
    if repeat:
        out["repeat"] = True
    return out


SEED = _int("seed", "Seed", 7, 0, 2**31 - 1,
            "The same seed is the same picture; a different seed is a different one.")
SIZE = _int("size", "Size (px)", 96, 16, 1024,
            "Sprites and tiles are drawn at this size, never resized down.")

GENERATORS = [
    {
        "id": "figures",
        "group": "Characters",
        "title": "Figure sprite sheet",
        "blurb": "One body, one gait, and the frames an engine plays. The far limb is a "
                 "depth cue, not a second leg.",
        "base": ["figures"],
        "out": "figures.png",
        "fields": [
            SEED,
            _choice("cycle", "Cycle", "walk", figure.cycles()),
            _int("count", "Frames", 8, 1, 24),
            _int("size", "Frame (px)", 96, 32, 256),
            _int("cols", "Columns", 8, 1, 12),
            _float("fps", "Frames / second", 12.0, 1.0, 60.0, step=1.0),
            _choice("facing", "Facing", "right", figure.FACINGS,
                    "right/left across the screen, down towards the camera, up away"),
        ],
    },
    {
        "id": "portrait",
        "group": "Characters",
        "title": "Portrait",
        "blurb": "A head as a stack of cross-sections: the outline is a support, not a "
                 "squeeze. Seed, style, hair and expression are independent.",
        "base": ["portrait"],
        "out": "portrait.png",
        "fields": [
            SEED,
            _int("size", "Size (px)", 384, 64, 1024),
            _choice("style", "Style", "", [""] + list(portrait_style.names()),
                    "empty = the seed's own style", optional=True),
            _choice("hair", "Hair", "", [""] + list(portrait_dna.HAIR_STYLES),
                    "empty = the seed's own hair", optional=True),
            _choice("expression", "Expression", "",
                    [""] + list(portrait_dna.EXPRESSIONS),
                    "empty = the seed's own expression", optional=True),
            _choice("sex", "Sex", "", [""] + list(portrait_demography.SEXES),
                    "empty = the seed's own; presentation, not a stereotype", optional=True),
            _choice("age", "Age", "", [""] + list(portrait_demography.AGES),
                    "empty = the seed's own; changes skin, hair and proportion", optional=True),
            _choice("climate", "Climate", "",
                    [""] + list(portrait_demography.CLIMATES),
                    "the lineage's adaptation, not a nationality", optional=True),
            _choice("culture", "Culture", "",
                    [""] + list(portrait_demography.CULTURES),
                    "the dress tradition: collar, trim and ornament", optional=True),
        ],
    },
    {
        "id": "portraits",
        "group": "Characters",
        "title": "Portrait sheet",
        "blurb": "A contact sheet of seeded portraits: one face per cell, the same "
                 "brush and the one light as every other asset.",
        "base": ["portraits"],
        "out": "portraits.png",
        "fields": [
            SEED,
            _int("count", "Portraits", 8, 1, 24),
            _int("size", "Cell (px)", 256, 64, 512),
            _int("cols", "Columns", 4, 1, 8),
            _choice("sex", "Pin sex", "", [""] + list(portrait_demography.SEXES),
                    "empty = the sheet samples the range", optional=True),
            _choice("age", "Pin age", "", [""] + list(portrait_demography.AGES),
                    "empty = the sheet samples the range", optional=True),
            _choice("climate", "Pin climate", "", [""] + list(portrait_demography.CLIMATES),
                    "empty = the sheet samples the range", optional=True),
            _choice("culture", "Pin culture", "", [""] + list(portrait_demography.CULTURES),
                    "empty = the sheet samples the range", optional=True),
        ],
    },
    {
        "id": "lineage",
        "group": "Characters",
        "title": "Portrait lineage",
        "blurb": "Two parents and their children: colours blend, shapes average and "
                 "drift, and the categories are drawn. A dynasty from genomes, not "
                 "from a face per character.",
        "base": ["lineage"],
        "out": "lineage.png",
        "fields": [
            SEED,
            _int("size", "Cell (px)", 208, 64, 512),
            _int("children", "Children", 6, 0, 12),
        ],
    },
    {
        "id": "arms",
        "group": "Heraldry",
        "title": "Coats of arms",
        "blurb": "Shields, fields, charges and the rule of tincture. A name is a seed; "
                 "the blazon is a view of the same spec.",
        "base": ["arms"],
        "out": "arms.png",
        "fields": [
            SEED,
            _int("count", "Coats", 12, 1, 32),
            _int("size", "Cell (px)", 190, 64, 512),
            _int("cols", "Columns", 4, 1, 8),
            _text("shield", "Force a shield", "", "empty = seeded per coat",
                  optional=True),
        ],
    },
    {
        "id": "blazon",
        "group": "Heraldry",
        "title": "One coat and its blazon",
        "blurb": "The proof of the round trip: one seeded coat, its English and "
                 "Russian blazon, and the tincture warnings, all from one spec.",
        "base": ["blazon"],
        "out": "blazon.png",
        "fields": [
            SEED,
            _text("shield", "Force a shield", "", "empty = the seed's own shield",
                  optional=True),
        ],
    },
    {
        "id": "flags",
        "group": "Heraldry",
        "title": "Flag sheet",
        "blurb": "Bands by projection, folds as a normal field, a pole and a fly. The "
                 "cloth is the same height field that lights it.",
        "base": ["flags"],
        "out": "flags.png",
        "fields": [
            SEED,
            _int("count", "Flags", 12, 1, 32),
            _int("size", "Cell (px)", 200, 64, 512),
            _int("cols", "Columns", 4, 1, 8),
            _text("ratio", "Proportion", "3:2", "e.g. 3:2 or 5:8"),
            _choice("shape", "Silhouette", "rectangle", SHAPES),
            _choice("hang", "Hung from", "pole", list(HANGINGS) + ["mixed", "none"]),
            _bool("flat", "Flat (hold nothing)", "--flat"),
        ],
    },
    {
        "id": "factions",
        "group": "Heraldry",
        "title": "Faction sheet",
        "blurb": "A faction from its name: the coat, the banner and the blazon, with "
                 "the livery that dresses the map and the icons.",
        "base": ["factions"],
        "out": "factions.png",
        "fields": [
            _int("seed", "Seed (labels only)", 7, 0, 2**31 - 1,
                 "the names decide the coat; the seed only labels the sheet"),
            _int("size", "Cell (px)", 168, 64, 512),
            _text("ratio", "Proportion", "3:2", "e.g. 3:2 or 5:8"),
            _text("name", "Names", "", "one per line or comma-separated; empty = demo set",
                  optional=True, repeat=True),
        ],
    },
    {
        "id": "anim",
        "group": "Heraldry",
        "title": "Flag animation",
        "blurb": "One flag through one loop: equal cells, a timeline and the phase each "
                 "frame was drawn at. Nothing is interpolated.",
        "base": ["anim"],
        "out": "anim.png",
        "fields": [
            SEED,
            _int("count", "Frames", 12, 4, 48),
            _int("size", "Frame (px)", 200, 64, 512),
            _int("cols", "Columns", 4, 1, 8),
            _float("fps", "Frames / second", 12.0, 1.0, 60.0, step=1.0),
            _int("padding", "Padding (px)", 2, 0, 16),
        ],
    },
    {
        "id": "maps",
        "group": "World",
        "title": "Strategic map",
        "blurb": "The wide view: terrain, relief, coastline, rivers, roads, provinces "
                 "and names, drawn in one raster pass and exported as data too.",
        "base": ["maps"],
        "out": "maps.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            SEED,
            _int("count", "Maps", 1, 1, 4),
            _int("cols", "Hexes across", 26, 8, 60),
            _int("rows", "Hexes down", 17, 6, 40),
            _float("size", "Hex radius (px)", 13.0, 5.0, 40.0, step=0.5),
            _float("sea", "Sea fraction", 0.42, 0.0, 0.90, step=0.01),
            _int("provinces", "Provinces", 8, 0, 30),
            _float("river", "River opacity", 0.92, 0.0, 1.0, step=0.01),
            _float("road", "Road opacity", 0.85, 0.0, 1.0, step=0.01),
            _int("seas", "Sea provinces", 8, 0, 30),
            _int("factions", "Realms", 4, 0, 12),
            _bool("no_names", "No names", "--no-names"),
        ],
    },
    {
        "id": "site",
        "group": "World",
        "title": "Site close-up",
        "blurb": "The camera down: the same hexes at a size where the ground is the "
                 "subject, with a carved river, cliffs and a wet shore.",
        "base": ["site"],
        "out": "site.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            SEED,
            _int("cols", "Hexes across", 8, 3, 16),
            _int("rows", "Hexes down", 6, 3, 12),
            _int("size", "Hex (px)", 56, 24, 128),
            _bool("flat", "Biome colour only", "--flat",
                  help="one panel, no ground material"),
            _float("river_fraction", "River fraction", 0.90, 0.50, 1.0, step=0.01,
                   help="the drainage quantile a river must beat"),
        ],
    },
    {
        "id": "iso",
        "group": "World",
        "title": "Isometric board",
        "blurb": "A board from the south: terraced hex prisms, world-sampled tops, "
                 "dirt cliffs, foam, and a hero who can face four ways.",
        "base": ["iso"],
        "out": "iso.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            SEED,
            _int("cols", "Hexes across", 11, 4, 20),
            _int("rows", "Hexes down", 8, 4, 16),
            _int("size", "Hex (px)", 46, 24, 96),
            _float("sea", "Sea fraction", 0.16, 0.0, 0.60, step=0.01),
            _float("rise", "Rise", 2.5, 0.0, 6.0, step=0.1,
                   help="height in hex sizes per unit of elevation"),
            _int("levels", "Terraces", 7, 0, 12),
            _text("hero", "Hero hex", "", "row,col; empty = nearest plain to the middle",
                  optional=True),
            _int("hero_seed", "Hero seed", 7, 0, 2**31 - 1),
            _choice("cycle", "Hero cycle", "idle", figure.cycles()),
            _choice("facing", "Hero facing", "right", figure.FACINGS),
        ],
    },
    {
        "id": "coast",
        "group": "World",
        "title": "Coast and border tiles",
        "blurb": "The sixty-four edge tiles a coastline is made of, plus the "
                 "biome-border panel and the measured count of which tiles the "
                 "demo maps actually wear.",
        "base": ["coast"],
        "out": "coast-tiles.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            SEED,
            _int("count", "Maps counted", 2, 1, 6),
            _float("size", "Tile radius (px)", 20.0, 8.0, 64.0, step=1.0),
            _float("hex", "Map hex radius", 13.0, 5.0, 40.0, step=0.5),
            _int("cols", "Map hexes across", 26, 8, 60),
            _int("rows", "Map hexes down", 17, 6, 40),
            _float("sea", "Map sea fraction", 0.42, 0.0, 0.90, step=0.01),
            _float("inland", "Border opacity", 0.55, 0.0, 1.0, step=0.01),
        ],
    },
    {
        "id": "grounds",
        "group": "Materials",
        "title": "Ground materials",
        "blurb": "Six materials from one painter and fifteen fields a row: mottle, "
                 "ripples, torus-cut facets and grit.",
        "base": ["grounds"],
        "out": "grounds.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            SEED,
            _int("size", "Cell (px)", 256, 64, 512),
            _bool("no_atlas", "No packed atlas", "--no-atlas"),
            _int("mips", "Mip levels", 1, 1, 8),
        ],
    },
    {
        "id": "tiles",
        "group": "Materials",
        "title": "Tiling textures",
        "blurb": "Truchet, Wang and reaction-diffusion, each seamless by construction, "
                 "with the seam error measured rather than promised.",
        "base": ["tiles"],
        "out": "tiles.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            SEED,
            _int("size", "Cell (px)", 256, 64, 512),
            _int("sim", "Simulation cells", 80, 16, 256),
            _int("steps", "Simulation steps", 5000, 100, 20000),
            _int("mips", "Mip levels", 1, 1, 8),
            _bool("no_atlas", "No packed atlas", "--no-atlas"),
        ],
    },
    {
        "id": "texture",
        "group": "Materials",
        "title": "Material sheet",
        "blurb": "The old core materials: fBm, marble, wood and stone, side by side.",
        "base": ["texture"],
        "out": "texture.png",
        "fields": [
            SEED,
            _int("size", "Cell (px)", 512, 64, 1024),
        ],
    },
    {
        "id": "icons",
        "group": "Interface",
        "title": "Icon set",
        "blurb": "Plates and charges, quantities as seven-segment numerals, tiers as "
                 "pips, a faction's livery on the plate and a packed atlas.",
        "base": ["icons"],
        "out": "icons.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            _int("seed", "Seed", 3, 0, 2**31 - 1),
            _int("count", "Icons", 8, 1, 24),
            _int("size", "Cell (px)", 192, 64, 512),
            _int("cols", "Columns", 9, 1, 12),
            _text("sizes", "Size sweep", "32,64,128", "comma-separated pixels"),
            _bool("no_atlas", "No packed atlas", "--no-atlas"),
            _text("name", "Dress in a faction", "",
                  "one name; empty = the kind's own hue", optional=True, repeat=True),
        ],
    },
    {
        "id": "ui",
        "group": "Interface",
        "title": "UI kit",
        "blurb": "Nine-slice frames, asymmetric insets, wood and stone bodies, and a "
                 "measured stretch error instead of a promise.",
        "base": ["ui"],
        "out": "ui.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            SEED,
            _int("count", "Seeded kits", 3, 0, 8),
            _int("scale", "Pixel density", 2, 1, 4),
            _choice("style", "Style", "", [""] + list(ui_style.names()),
                    "empty = every named style", optional=True),
            _bool("svg", "Also write SVG", "--svg"),
        ],
    },
    {
        "id": "text",
        "group": "Interface",
        "title": "Type specimen",
        "blurb": "The generated face: an alphabet of skeletons, a pen for weight, "
                 "contrast, serifs and slant, and the metrics an engine needs.",
        "base": ["text"],
        "out": "text.png",
        "slow": True,
        "timeout": 240,
        "fields": [
            _int("size", "Pixels / em", 20, 8, 64),
            _int("width", "Page width (px)", 1400, 600, 3000),
            _choice("face", "Face", "", [""] + list(text_pen.names()),
                    "empty = all five; one name for one face", optional=True),
        ],
    },
    {
        "id": "palette",
        "group": "Core",
        "title": "Palette",
        "blurb": "A seeded set of swatches in OKLCH, resolved to sRGB, with the numbers "
                 "beside the colours.",
        "base": ["palette"],
        "out": "palette.png",
        "fields": [
            SEED,
            _int("count", "Colours", 6, 1, 24),
        ],
    },
    {
        "id": "shapes",
        "group": "Core",
        "title": "SDF reference",
        "blurb": "The primitives every other generator draws with: circles, boxes, "
                 "capsules, locks and the smooth unions between them.",
        "base": ["shapes"],
        "out": "shapes.png",
        "fields": [
            _int("size", "Cell (px)", 160, 32, 512),
            _int("cols", "Columns", 3, 1, 8),
        ],
    },
]


BY_ID = {g["id"]: g for g in GENERATORS}


def public() -> list:
    """The catalog as the browser needs it: no argv, no timeouts, no filenames."""
    keep = ("id", "group", "title", "blurb", "slow", "fields")
    fields_keep = ("name", "label", "type", "default", "min", "max", "step",
                   "choices", "help", "optional", "repeat")
    out = []
    for gen in GENERATORS:
        item = {k: gen[k] for k in keep if k in gen}
        item["fields"] = [{k: f[k] for k in fields_keep if k in f}
                          for f in gen["fields"]]
        out.append(item)
    return out
