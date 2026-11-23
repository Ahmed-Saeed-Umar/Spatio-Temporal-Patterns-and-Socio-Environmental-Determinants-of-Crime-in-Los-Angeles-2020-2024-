"""Rule-based grouping of the ~140 LAPD crime descriptions into broader categories.

Why rules and not a language model?
    * There are only ~140 distinct descriptions, so a transparent keyword mapping is
      quick to audit (every rule is visible below) and fully reproducible.
    * It has no extra dependency and gives the same result on every machine.

Rules are checked in order; the FIRST match wins, so order matters
(e.g. "ASSAULT WITH DEADLY WEAPON" and "CRIMINAL THREATS - NO WEAPON DISPLAYED" must
be caught by the Assault / Threats rules before the Weapons rule).
"""
import re

import pandas as pd

# (group name, regex searched in the upper-case crime description)
RULES = [
    ("Crimes Against Children", r"CHILD|CRM AGNST CHLD|DRUGS, TO A MINOR"),
    ("Homicide & Kidnapping", r"HOMICIDE|MANSLAUGHTER|LYNCHING|KIDNAPPING|FALSE IMPRISONMENT"
                              r"|HUMAN TRAFFICKING - INVOLUNTARY"),
    ("Sex Crimes", r"RAPE|SEX|SODOMY|ORAL COPULATION|LEWD|INDECENT EXPOSURE|PEEPING|PIMPING"
                   r"|PANDERING|INCEST|BEASTIALITY|HUMAN TRAFFICKING - COMMERCIAL"),
    ("Assault & Battery", r"ASSAULT|BATTERY|INTIMATE PARTNER"),
    ("Threats & Court Orders", r"THREAT|RESTRAINING ORDER|COURT ORDER|CONTEMPT|STALKING"),
    ("Weapons & Firearms", r"FIREARM|SHOTS FIRED|DISCHARGE|WEAPON|BRANDISH|BOMB|REPLICA"),
    ("Robbery & Extortion", r"ROBBERY|EXTORTION"),
    ("Vehicle Theft & Break-ins", r"^VEHICLE|MOTOR VEHICLE|FROM VEHICLE|DWOC"
                                  r"|DRIVING WITHOUT OWNER|^BOAT"),
    ("Burglary", r"^BURGLARY"),
    ("Vandalism & Arson", r"VANDALISM|ARSON|TELEPHONE PROPERTY"),
    ("Fraud & Identity Theft", r"IDENTITY|BUNCO|CREDIT CARD|FORGERY|COUNTERFEIT|DOCUMENT WORTHLESS"
                               r"|COMPUTER|INNKEEPER|INSURANCE FRAUD|BRIBERY|EMBEZZLEMENT"
                               r"|DISHONEST EMPLOYEE"),
    ("Theft & Larceny", r"THEFT|SHOPLIFTING|PICKPOCKET|PURSE SNATCHING|BIKE|TILL TAP|DRUNK ROLL"
                        r"|COIN MACHINE"),
]
OTHER = "Other / Public Order"

# Broader 4-way class used for headline charts
CRIME_CLASS = {
    "Crimes Against Children": "Violent",
    "Homicide & Kidnapping": "Violent",
    "Sex Crimes": "Violent",
    "Assault & Battery": "Violent",
    "Weapons & Firearms": "Violent",
    "Robbery & Extortion": "Violent",
    "Threats & Court Orders": "Violent",
    "Vehicle Theft & Break-ins": "Property",
    "Burglary": "Property",
    "Vandalism & Arson": "Property",
    "Theft & Larceny": "Property",
    "Fraud & Identity Theft": "Fraud",
    OTHER: "Other",
}

_COMPILED = [(name, re.compile(rx)) for name, rx in RULES]


def group_crime(description: str) -> str:
    """Return the crime group for one crime description."""
    text = str(description).upper()
    for name, rx in _COMPILED:
        if rx.search(text):
            return name
    return OTHER


def build_group_map(descriptions) -> dict:
    """Map every unique description to its group (fast: ~140 regex calls, not 1M)."""
    return {d: group_crime(d) for d in pd.unique(descriptions)}
