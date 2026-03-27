# sectors.py — Asset groupings for the interactive selection menu.
# Add or reorganise sectors/assets here as your universe grows.

SECTORS: dict[str, list[str]] = {
    "US Equities":       ["ES", "NQ", "YM", "RTY"],
    "European Equities": ["FDAX", "FESX"],
    "US Bonds":          ["ZN"],
    "European Bonds":    ["FGBL", "FGBM", "FGBS", "FBTP"],
    "FX":                ["6A", "6B", "6E", "6J"],
    "Energy":            ["CL", "BRN", "NG", "HO"],
    "Metals":            ["GC", "SI", "HG", "G"],
    "Grains":            ["ZC", "ZS", "ZW", "ZL", "ZM", "C", "W"],
    "Softs":             ["KC", "CC", "SB", "CT"],
    "Livestock":         ["LE", "HE", "GF"],
    "STIR":              ["SO3-7th", "I-7th"],
}
