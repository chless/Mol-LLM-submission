homo = [
    "I would like to know the highest occupied molecular orbital (HOMO) energy of this molecule, could you please provide it?",
    "Please provide the HOMO energy value for this molecule.",
    "I am interested in the HOMO energy of this molecule, could you tell me what it is?",
    "What is the highest occupied molecular orbital (HOMO) energy of this molecule?",
    "Could you give me the HOMO energy value of this molecule?",
    "What is the HOMO energy of this molecule?",
    "Please provide the highest occupied molecular orbital (HOMO) energy value for this molecule.",
    "Please provide me with the HOMO energy value of this molecule.",
    "What is the HOMO level of energy for this molecule?",
    "I would like to know the HOMO energy of this molecule, could you please provide it?",
    "Can you tell me the value of the HOMO energy for this molecule?",
    "Please provide the highest occupied molecular orbital (HOMO) energy of this molecule.",
]

lumo = [
    "Please provide me with the LUMO energy value of this molecule.",
    "I am interested in the LUMO energy of this molecule, could you tell me what it is?",
    "I would like to know the lowest unoccupied molecular orbital (LUMO) energy of this molecule, could you please provide it?",
    "What is the LUMO energy of this molecule?",
    "What is the LUMO level of energy for this molecule?",
    "I would like to know the LUMO energy of this molecule, could you please provide it?",
    "What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?",
    "Could you give me the LUMO energy value of this molecule?",
    "Please provide the lowest unoccupied molecular orbital (LUMO) energy value for this molecule.",
    "Please provide the lowest unoccupied molecular orbital (LUMO) energy of this molecule.",
    "Can you tell me the value of the LUMO energy for this molecule?",
    "Please provide the LUMO energy value for this molecule.",
]

homo_lumo_gap = [
    "Please provide the gap between HOMO and LUMO of this molecule.",
    "I would like to know the HOMO-LUMO gap of this molecule, can you provide it?",
    "Please give me the HOMO-LUMO gap energy for this molecule.",
    "Can you give me the energy difference between the HOMO and LUMO orbitals of this molecule?",
    "Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.",
    "I need to know the HOMO-LUMO gap energy of this molecule, could you please provide it?",
    "What is the energy separation between the HOMO and LUMO of this molecule?",
    "Could you tell me the energy difference between HOMO and LUMO for this molecule?",
    "What is the HOMO-LUMO gap of this molecule?",
]

esol = [
    "Please provide the aqueous solubility of this molecule.",
    "I would like to know the solubility of this molecule in water, can you provide it?",
    "Please give me the aqueous solubility value for this molecule.",
    "Can you give me the solubility of this molecule in water?",
    "Please provide the solubility of this molecule in an aqueous environment.",
    "I need to know the solubility of this molecule in water, could you please provide it?",
    "What is the aqueous solubility of this molecule?",
    "Could you tell me the solubility of this molecule in water?",
    "What is the water solubility of this molecule?",
]

lipo = [
    "Please provide the distribution coefficient (logD) of this molecule in octanol and water.",
    "I would like to know the logD value of this molecule in octanol and water, can you provide it?",
    "Please give me the logD (octanol/water partition coefficient) for this molecule.",
    "Can you give me the distribution coefficient (logD) of this molecule in octanol and water?",
    "Please provide the logD value for this molecule in an octanol-water system.",
    "I need to know the logD (octanol/water partition coefficient) of this molecule, could you please provide it?",
    "What is the logD value of this molecule in octanol and water?",
    "Could you tell me the logD (distribution coefficient) for this molecule in octanol and water?",
    "What is the octanol/water distribution coefficient (logD) of this molecule?",
]

bace = [
    "Please provide the biological activity of this molecule against BACE-1.",
    "I would like to know the activity of this molecule against the BACE-1 enzyme, can you provide it?",
    "Please give me the BACE-1 inhibitory activity for this molecule.",
    "Can you give me the biological activity data of this molecule against BACE-1?",
    "Please provide the inhibitory activity of this molecule against the BACE-1 enzyme.",
    "I need to know the activity of this molecule against BACE-1, could you please provide it?",
    "What is the BACE-1 inhibitory activity of this molecule?",
    "Could you tell me the activity of this molecule against the BACE-1 enzyme?",
    "What is the biological activity of this molecule against BACE-1?",
]

bbbp = [
    "Please provide the blood-brain barrier penetration ability of this molecule.",
    "I would like to know if this molecule can penetrate the blood-brain barrier, can you provide this information?",
    "Please give me the blood-brain barrier permeability of this molecule.",
    "Can you give me the information on whether this molecule can cross the blood-brain barrier?",
    "Please predict the blood-brain barrier penetration for this molecule.",
    "I need to know the ability of this molecule to penetrate the blood-brain barrier, could you please provide it?",
    "What is the blood-brain barrier penetration capability of this molecule?",
    "Could you tell me if this molecule can penetrate the blood-brain barrier?",
    "What is the blood-brain barrier permeability status of this molecule?",
]

clintox_FDA_APPROVED = [
    "Please provide the FDA approval status of this drug.",
    "I would like to know if this drug is approved by the FDA, can you provide this information?",
    "Please tell me whether this drug has FDA approval.",
    "Can you give me the FDA approval status for this drug?",
    "Please provide information on whether this drug has been approved by the FDA.",
    "I need to know if this drug is FDA-approved, could you please provide it?",
    "What is the FDA approval status of this drug?",
    "Could you tell me if this drug is approved by the FDA?",
    "Is this drug FDA-approved?",
    "Does this drug have FDA approval?",
]

clintox_CT_TOX = [
    "Please provide information on whether this drug passed clinical trials.",
    "I would like to know if this drug passed clinical trials, can you provide this information?",
    "Please tell me whether the clinical trials for this drug were successful.",
    "Can you give me the pass/fail status of the clinical trials for this drug?",
    "Please provide details on whether this drug successfully passed clinical trials.",
    "I need to know if this drug passed clinical trials, could you please provide it?",
    "What is the clinical trial success status of this drug?",
    "Could you tell me if this drug passed its clinical trials?",
    "Did this drug pass clinical trials?",
    "Is this drug successful for clinical trials?",
]

toxcast = [
    "Please provide information on whether this molecule is toxic considering the ACEA_T47D_80hr_Negative assay result.",
    "I would like to know if this molecule shows toxicity based on the ACEA_T47D_80hr_Negative bioassay, can you provide this information?",
    "Please tell me whether this molecule is toxic according to the ACEA_T47D_80hr_Negative assay results.",
    "Can you give me the toxicity status of this molecule with respect to the ACEA_T47D_80hr_Negative bioassay?",
    "Please provide details on whether this molecule is toxic based on ACEA_T47D_80hr_Negative assay results.",
    "I need to know if this molecule is toxic according to the ACEA_T47D_80hr_Negative bioassay, could you please provide it?",
    "Does this molecule exhibit toxicity in the ACEA_T47D_80hr_Negative bioassay?",
    "Could you tell me if this molecule is toxic based on the ACEA_T47D_80hr_Negative assay results?",
    "Is this molecule associated with toxicity in the ACEA_T47D_80hr_Negative bioassay?",
    "What is the toxicity status of this molecule with respect to the ACEA_T47D_80hr_Negative assay?",
]

sider = [
    "Please provide information on whether this molecule has side effects.",
    "I would like to know if this molecule has side effects, can you provide this information?",
    "Please tell me whether this molecule causes any side effects.",
    "Can you give me the side effect status for this molecule?",
    "Please predict whether this molecule has associated with side effects.",
    "I need to know if this molecule has side effects, could you please provide it?",
    "Does this molecule have known side effects?",
    "Could you tell me if this molecule has side effects?",
    "Is this molecule associated with side effects?",
]

tox21 = [
    "Please provide information on whether this molecule is toxic based on nuclear receptor signaling bioassays.",
    "I would like to know if this molecule shows toxicity in nuclear receptor signaling bioassays, can you provide this information?",
    "Please tell me whether this molecule is toxic according to nuclear receptor signaling bioassays results.",
    "Can you give me the toxicity status of this molecule with respect to nuclear receptor signaling bioassays?",
    "Please provide details on whether this molecule is toxic based on nuclear receptor signaling bioassays results.",
    "I need to know if this molecule is toxic according to nuclear receptor signaling bioassays, could you please provide it?",
    "Does this molecule exhibit toxicity in nuclear receptor signaling bioassays?",
    "Could you tell me if this molecule is toxic based on nuclear receptor signaling bioassays results?",
    "Is this molecule associated with toxicity in nuclear receptor signaling bioassays?",
    "What is the toxicity status of this molecule with respect to nuclear receptor signaling bioassays?",
]
hiv = [
    "Please provide information on whether this molecule inhibits HIV replication.",
    "I would like to know if this molecule can inhibit HIV replication, can you provide this information?",
    "Please tell me whether this molecule prevents HIV replication.",
    "Can you give me the inhibition status of this molecule with respect to HIV replication?",
    "Please provide details on whether this molecule inhibits HIV replication.",
    "I need to know if this molecule is capable of inhibiting HIV replication, could you please provide it?",
    "Does this molecule exhibit HIV replication inhibition?",
    "Could you tell me if this molecule inhibits HIV replication?",
    "Is this molecule associated with the inhibition of HIV replication?",
    "What is the HIV replication inhibition status of this molecule?",
]

chebi_20_mol2text = [
    "Can you describe the molecular structure of this compound?",
    "Please provide a detailed description of the molecular structure.",
    "I would like to know the structural features of this molecule, could you describe them?",
    "What is the molecular structure of this compound?",
    "Could you give me a description focusing on the molecular structure?",
    "What are the key structural characteristics of this molecule?",
    "Please provide a comprehensive overview of the molecular structure.",
    "Could you tell me about the arrangement of atoms in this molecule?",
    "I am interested in the molecular structure, can you describe it?",
    "Please provide me with a detailed structural description of this molecule.",
    "What is the structural layout and geometry of this molecule?",
    "Can you give me an overview of the molecular structure?",
]

chebi_20_text2mol = [
    "Can you create a molecule based on this structural description?",
    "Please generate a molecule that matches the given molecular structure.",
    "I need a molecule constructed from this description of its structure.",
    "What molecule can be formed from the following structural details?",
    "Could you build a molecule based on these structural characteristics?",
    "Create a molecular model that fits this structural description.",
    "Please construct a molecule using the provided structure details.",
    "Can you form a molecule according to this description of its structure?",
    "I am interested in a molecule generated from these structural features.",
    "Please design a molecule that corresponds to this structural overview.",
    "What is the molecule that can be derived from these structural description?",
    "Can you synthesize a molecule based on this structural information?",
]
