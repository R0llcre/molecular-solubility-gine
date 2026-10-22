from __future__ import annotations


MOTIF_SMARTS = {
    "hydroxyl": "[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]([#6])[#6]",
    "tertiary_amine": "[NX3;H0;!$(NC=O)]([#6])([#6])[#6]",
    "carboxyl": "C(=O)[OX2H1]",
    "ester": "C(=O)O[#6]",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "amide": "C(=O)N",
    "nitro": "[NX3](=O)=O",
    "sulfonamide": "S(=O)(=O)N",
    "thiol": "[SX2H]",
    "sulfide": "[SX2]([#6])[#6]",
    "phosphate": "P(=O)(O)(O)",
    "halogen": "[F,Cl,Br,I]",
    "aromatic_hetero": "[a;!c]",
}
