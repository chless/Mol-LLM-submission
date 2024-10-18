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

smol_molecule_captioning = [
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

smol_molecule_generation = [
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


smol_name_conversion_s2f = [
    # '<INPUT> is the SMILES representation of a molecule. What is its molecular formula?',
    # 'Convert the SMILES representation of a molecule <INPUT> into molecular formula.',
    '<INPUT> is the SELFIES representation of a molecule. What is its molecular formula?',
    'Convert the SELFIES representation of a molecule <INPUT> into molecular formula.',
    'What is the formula of the molecule <INPUT> ?',
    'Can you give the molecular molecular formula of <INPUT> ?',
    'Please write the molecular formula of the molecule <INPUT> .',
    # 'Given the SMILES representation <INPUT>, what would be its molecular formula?',
    # 'The SMILES representation <INPUT> represents a specific molecule. Can you reveal its molecular formula?',
    # 'Considering the SMILES code <INPUT>, can you determine the corresponding molecular formula?',
    'Given the SELFIES representation <INPUT>, what would be its molecular formula?',
    'The SELFIES representation <INPUT> represents a specific molecule. Can you reveal its molecular formula?',
    'Considering the SELFIES code <INPUT>, can you determine the corresponding molecular formula?',
    'Can you tell me the molecular formula of <INPUT> ?',
    "I'd like to know the molecular formula of <INPUT> . Can you tell me?",
    "What is the molecular formula for the molecule denoted by <INPUT> ?",
    "What is the molecular formula of <INPUT> ?",
    "Please provide the molecular formula for <INPUT> .",
]
smol_name_conversion_s2i = [
    # '<INPUT> is the SMILES representation of a molecule. What is its IUPAC name?',
    # 'Convert the SMILES representation of a molecule <INPUT> into IUPAC name.',
    '<INPUT> is the SELFIES representation of a molecule. What is its IUPAC name?',
    'Convert the SELFIES representation of a molecule <INPUT> into IUPAC name.',
    'What is the IUPAC name of the molecule <INPUT> ?',
    'Can you give the IUPAC name of the molecule <INPUT> ?',
    'Please write the IUPAC name of the molecule <INPUT> .',
    # '<INPUT> The above is a SMILES representation. Write the IUPAC name of the corresponding molecule.',
    # 'Determine the IUPAC name for the molecule represented by the following SMILES representation: <INPUT> .',
    # 'What is the IUPAC name for the molecule whose SMILES representation is <INPUT> ?',
    '<INPUT> The above is a SELFIES representation. Write the IUPAC name of the corresponding molecule.',
    'Determine the IUPAC name for the molecule represented by the following SELFIES representation: <INPUT> .',
    'What is the IUPAC name for the molecule whose SELFIES representation is <INPUT> ?',
    'Determine the IUPAC name for the molecule denoted by <INPUT> .',
    # 'Translate the given SMILES formula of a molecule <INPUT> into its IUPAC name.',
    'Translate the given SELFIES formula of a molecule <INPUT> into its IUPAC name.',
    'Provide the IUPAC name for the molecule represented as <INPUT> .',
    # 'Convert the following SMILES notation <INPUT> into its IUPAC nomenclature.',
    # 'Turn the given SMILES symbol of a molecule <INPUT> into its respective IUPAC name.'
    'Convert the following SELFIES notation <INPUT> into its IUPAC nomenclature.',
    'Turn the given SELFIES symbol of a molecule <INPUT> into its respective IUPAC name.'
]
smol_name_conversion_i2s = [
    # '<INPUT> is the IUPAC name of a molecule. Please give its SMILES representation.',
    # 'Convert the IUPAC name of a molecule <INPUT> into SMILES representation.',
    # 'What is the SMILES representation of the molecule with IUPAC name <INPUT> ?',
    # 'Can you give the SMILES notation of the molecule <INPUT> ?',
    # 'Please write the SMILES representation of the molecule <INPUT> .',
    # '<INPUT> The above is the IUPAC name of a molecule. Write its SMILES notation.',
    # 'The IUPAC name of a certain molecule is <INPUT> . Can you provide its SMILES representation?',
    # 'Please identify the SMILES representation of the molecule named <INPUT> .',
    # 'For the molecule with <INPUT> as the IUPAC name, what is the corresponding SMILES representation?',
    # 'What is the SMILES notation for <INPUT> ?',
    # 'Could you provide the SMILES for <INPUT> ?',
    # 'Can you tell me the SMILES representation for the molecule <INPUT> ?',
    # 'What is the SMILES representation for <INPUT> ?'
    '<INPUT> is the IUPAC name of a molecule. Please give its SELFIES representation.',
    'Convert the IUPAC name of a molecule <INPUT> into SELFIES representation.',
    'What is the SELFIES representation of the molecule with IUPAC name <INPUT> ?',
    'Can you give the SELFIES notation of the molecule <INPUT> ?',
    'Please write the SELFIES representation of the molecule <INPUT> .',
    '<INPUT> The above is the IUPAC name of a molecule. Write its SELFIES notation.',
    'The IUPAC name of a certain molecule is <INPUT> . Can you provide its SELFIES representation?',
    'Please identify the SELFIES representation of the molecule named <INPUT> .',
    'For the molecule with <INPUT> as the IUPAC name, what is the corresponding SELFIES representation?',
    'What is the SELFIES notation for <INPUT> ?',
    'Could you provide the SELFIES for <INPUT> ?',
    'Can you tell me the SELFIES representation for the molecule <INPUT> ?',
    'What is the SELFIES representation for <INPUT> ?'
]
smol_name_conversion_i2f = [
    "<INPUT> is the IUPAC name of a molecule. Please give its molecular formula.",
    "Convert the IUPAC name of a molecule <INPUT> into molecular formula.",
    "What is the molecular formula of the molecule <INPUT> ?",
    "Can you give the molecular formula of the molecule <INPUT> ?",
    "Please write the molecular formula of the molecule <INPUT> .",
    "<INPUT> The above is the IUPAC name of a molecule. Write its molecular formula.",
    "What is the molecular formula of the molecule identified by <INPUT> ?",
    "For the molecule named <INPUT> by IUPAC nomenclature, provide its molecular formula.",
    "Name the molecular formula for the molecule established as <INPUT> using the IUPAC nomenclature.",
    "Under IUPAC nomenclature, a molecule named <INPUT> is given. What is its corresponding molecular formula?",
    "Translate the chemical IUPAC name <INPUT> into its molecular formula.",
    "Provide the molecular formula for the IUPAC named substance <INPUT> .",
    "What is the molecular formula of the compound with this IUPAC name <INPUT> ?",
]

smol_forward_synthesis = [
    "Can you create a molecule based on this reactant?",
    "Please generate the product molecule that results from the given reactant.",
    "I need a molecule formed from this reactant.",
    "What molecule can be formed from the following reactant?",
    "Could you build a molecule based on this reactant?",
    "Create a molecular model that is the product of this reactant.",
    "Please construct the product molecule using the provided reactant.",
    "Can you form a molecule according to this reactant?",
    "I am interested in the molecule generated from this reactant.",
    "Please design a molecule that corresponds to the transformation of this reactant.",
    "What is the molecule that can be derived from this reactant?",
    "Can you synthesize a molecule based on this reactant?",
]

smol_retrosynthesis = [
    "Can you identify a reactant that can produce this molecule?",
    "Please generate the reactant molecule that could lead to the given product.",
    "I need a reactant molecule that forms this product.",
    "What reactant can be used to form the following product?",
    "Could you propose a reactant based on this product?",
    "Create a molecular model of a reactant that yields this product.",
    "Please suggest a reactant molecule that could synthesize the provided product.",
    "Can you determine a reactant according to this product?",
    "I am interested in a reactant that can generate this molecule.",
    "Please design a reactant molecule that transforms into this product.",
    "What is the reactant from which this molecule can be derived?",
    "Can you propose a reactant to synthesize this molecule?",
]

smol_molecule_generation = [
    # smol, if want to use, need to change the stage3_dm.py
    # 'Based on the given information, generate a molecule that meets the desired specifications: <INPUT>',
    # 'Give me a molecule that satisfies the conditions outlined in the description: <INPUT>',
    # 'Generate a molecule based on this description: <INPUT>',
    # 'Can you create a molecule that matches the given characteristics? <INPUT>',
    # 'I need a molecule that meets the following conditions: <INPUT> Please represent the molecule in SMILES.',
    # 'Suppose there is a molecule that meets the following description: <INPUT> Please write the SMILES representation of it.',
    # '<INPUT> Use the above information to create a molecule.',
    # 'Build a molecule that meets the requirement: <INPUT>',
    # 'Generate a molecule that fulfills the requirement: <INPUT>',
    # 'Conceptualize a molecule that meets the specified attribute(s): <INPUT>',
    # 'Come up with a molecule based on the description: <INPUT>',
    # 'Could you please return a molecule that adheres to this description? <INPUT>',
    # 'I give you a description of a molecule, and you need to return one molecule in SMILES that meets the description. The description: <INPUT>'
    
    # mol-llm
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

smol_molecule_captioning = [
    # smol, if want to use, need to change the stage3_dm.py
    # 'Could you give me a brief introduction to this molecule? <INPUT>',
    # 'Describe this molecule: <INPUT>',
    # 'Please give me some details about this molecule. <INPUT>',
    # 'Please provide a brief introduction to this molecule. <INPUT>',
    # 'Tell me something about this molecule: <INPUT>',
    # '<INPUT> The above is a compound. Could you please tell me something about it?',
    # '<INPUT> What do you know about the molecule?',
    # 'Here is a molecule represented with SMILE: <INPUT> . Please describe it in natural language.',
    # 'Can you briefly describe the molecular encoded by this SMILES notation? <INPUT>',
    # 'I need a brief explanation of the molecule denoted in this SMILES notation. <INPUT>',
    # "I'd like a short overview about this molecule. Can you do that? <INPUT>",
    # 'May I have a capsulized explanation for this molecule? <INPUT>',
    # 'What can you tell me about this molecule? <INPUT>'
    
    # mol-llm
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
]

qm9_dipole_moment = [
    "I would like to know the dipole moment of this molecule, could you please provide it?",
    "Please provide the dipole moment value for this molecule.",
    "I am interested in the dipole moment of this molecule, could you tell me what it is?",
    "What is the dipole moment of this molecule?",
    "Could you give me the dipole moment value of this molecule?",
    "What is the value of the dipole moment for this molecule?",
    "Please provide the dipole moment value for this molecule.",
    "Please provide me with the dipole moment of this molecule.",
    "What is the dipole moment measurement for this molecule?",
    "I would like to know the dipole moment of this molecule, could you please provide it?",
    "Can you tell me the value of the dipole moment for this molecule?",
    "Please provide the dipole moment of this molecule.",
]

qm9_isotropic_polarizability = [
    "I would like to know the isotropic polarizability of this molecule, could you please provide it?",
    "Please provide the isotropic polarizability value for this molecule.",
    "I am interested in the isotropic polarizability of this molecule, could you tell me what it is?",
    "What is the isotropic polarizability of this molecule?",
    "Could you give me the isotropic polarizability value of this molecule?",
    "What is the isotropic polarizability of this molecule?",
    "Please provide the isotropic polarizability value for this molecule.",
    "Please provide me with the isotropic polarizability of this molecule.",
    "What is the isotropic polarizability measurement for this molecule?",
    "I would like to know the isotropic polarizability of this molecule, could you please provide it?",
    "Can you tell me the value of the isotropic polarizability for this molecule?",
    "Please provide the isotropic polarizability of this molecule.",
]

qm9_electronic_spatial_extent = [
    "I would like to know the electronic spatial extent of this molecule, could you please provide it?",
    "Please provide the electronic spatial extent value for this molecule.",
    "I am interested in the electronic spatial extent of this molecule, could you tell me what it is?",
    "What is the electronic spatial extent of this molecule?",
    "Could you give me the electronic spatial extent value of this molecule?",
    "What is the electronic spatial extent of this molecule?",
    "Please provide the electronic spatial extent value for this molecule.",
    "Please provide me with the electronic spatial extent of this molecule.",
    "What is the electronic spatial extent measurement for this molecule?",
    "I would like to know the electronic spatial extent of this molecule, could you please provide it?",
    "Can you tell me the value of the electronic spatial extent for this molecule?",
    "Please provide the electronic spatial extent of this molecule.",
]

qm9_zero_point_vibrational_energy = [
    "I would like to know the zero point vibrational energy of this molecule, could you please provide it?",
    "Please provide the zero point vibrational energy value for this molecule.",
    "I am interested in the zero point vibrational energy of this molecule, could you tell me what it is?",
    "What is the zero point vibrational energy of this molecule?",
    "Could you give me the zero point vibrational energy value of this molecule?",
    "What is the zero point vibrational energy of this molecule?",
    "Please provide the zero point vibrational energy value for this molecule.",
    "Please provide me with the zero point vibrational energy of this molecule.",
    "What is the zero point vibrational energy measurement for this molecule?",
    "I would like to know the zero point vibrational energy of this molecule, could you please provide it?",
    "Can you tell me the value of the zero point vibrational energy for this molecule?",
    "Please provide the zero point vibrational energy of this molecule.",
]

qm9_heat_capacity_298K = [
    "I would like to know the heat capacity of this molecule at 298.15K, could you please provide it?",
    "Please provide the heat capacity value at 298.15K for this molecule.",
    "I am interested in the heat capacity of this molecule at 298.15K, could you tell me what it is?",
    "What is the heat capacity of this molecule at 298.15K?",
    "Could you give me the heat capacity value of this molecule at 298.15K?",
    "What is the heat capacity at 298.15K of this molecule?",
    "Please provide the heat capacity value at 298.15K for this molecule.",
    "Please provide me with the heat capacity of this molecule at 298.15K.",
    "What is the heat capacity measurement at 298.15K for this molecule?",
    "I would like to know the heat capacity at 298.15K of this molecule, could you please provide it?",
    "Can you tell me the value of the heat capacity for this molecule at 298.15K?",
    "Please provide the heat capacity of this molecule at 298.15K.",
]

qm9_internal_energy_298K = [
    "I would like to know the internal energy of this molecule at 298.15K, could you please provide it?",
    "Please provide the internal energy value at 298.15K for this molecule.",
    "I am interested in the internal energy of this molecule at 298.15K, could you tell me what it is?",
    "What is the internal energy of this molecule at 298.15K?",
    "Could you give me the internal energy value of this molecule at 298.15K?",
    "What is the internal energy at 298.15K of this molecule?",
    "Please provide the internal energy value at 298.15K for this molecule.",
    "Please provide me with the internal energy of this molecule at 298.15K.",
    "What is the internal energy measurement at 298.15K for this molecule?",
    "I would like to know the internal energy at 298.15K of this molecule, could you please provide it?",
    "Can you tell me the value of the internal energy for this molecule at 298.15K?",
    "Please provide the internal energy of this molecule at 298.15K.",
]

qm9_enthalpy_298K = [
    "I would like to know the enthalpy of this molecule at 298.15K, could you please provide it?",
    "Please provide the enthalpy value at 298.15K for this molecule.",
    "I am interested in the enthalpy of this molecule at 298.15K, could you tell me what it is?",
    "What is the enthalpy of this molecule at 298.15K?",
    "Could you give me the enthalpy value of this molecule at 298.15K?",
    "What is the enthalpy at 298.15K of this molecule?",
    "Please provide the enthalpy value at 298.15K for this molecule.",
    "Please provide me with the enthalpy of this molecule at 298.15K.",
    "What is the enthalpy measurement at 298.15K for this molecule?",
    "I would like to know the enthalpy at 298.15K of this molecule, could you please provide it?",
    "Can you tell me the value of the enthalpy for this molecule at 298.15K?",
    "Please provide the enthalpy of this molecule at 298.15K.",
]

qm9_free_energy_298K = [
    "I would like to know the free energy of this molecule at 298.15K, could you please provide it?",
    "Please provide the free energy value at 298.15K for this molecule.",
    "I am interested in the free energy of this molecule at 298.15K, could you tell me what it is?",
    "What is the free energy of this molecule at 298.15K?",
    "Could you give me the free energy value of this molecule at 298.15K?",
    "What is the free energy at 298.15K of this molecule?",
    "Please provide the free energy value at 298.15K for this molecule.",
    "Please provide me with the free energy of this molecule at 298.15K.",
    "What is the free energy measurement at 298.15K for this molecule?",
    "I would like to know the free energy at 298.15K of this molecule, could you please provide it?",
    "Can you tell me the value of the free energy for this molecule at 298.15K?",
    "Please provide the free energy of this molecule at 298.15K.",
]
