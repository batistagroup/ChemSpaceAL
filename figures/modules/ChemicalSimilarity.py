from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit.Chem import Descriptors

imatinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
nilotinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)C(=O)Nc4cc(cc(c4)n5cc(nc5)C)C(F)(F)F"
dasatinib = "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl"
bosutinib = "Clc1c(OC)cc(c(Cl)c1)Nc4c(C#N)cnc3cc(OCCCN2CCN(CC2)C)c(OC)cc34"
ponatinib = "Cc1ccc(cc1C#Cc2cnc3n2nccc3)C(=O)Nc4ccc(c(c4)C(F)(F)F)CN5CCN(CC5)C"
bafetinib = "CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)CN3CC[C@@H](C3)N(C)C)C(F)(F)F)NC4=NC=CC(=N4)C5=CN=CN=C5"
scores = {
    "complex0": 64.5,
    "complex1": 55.0,
    "complex2": 37.0,
    "complex3": 42.0,
    "complex4": 58.5,
    "complex5": 64.5,
}
binders = [imatinib, nilotinib, dasatinib, bosutinib, ponatinib, bafetinib]
