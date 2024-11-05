FS = [
    '<INPUT> Based on the reactants and reagents given above, suggest a possible product.',
    'Based on the given reactants and reagents: <INPUT>, what product could potentially be produced?',
    'Given the following reactants and reagents, please provide a possible product. <INPUT>',
    '<INPUT> Given the above reactants and reagents, what could be a probable product of their reaction?',
    'Please provide a feasible product that could be formed using these reactants and reagents: <INPUT> .',
    'Consider that for a chemical reaction, if <INPUT> is/are the reactants and reagents, what can be the product?',
    'Propose a potential product given these reactants and reagents. <INPUT>',
    'Predict the product of a chemical reaction with <INPUT> as the reactants and reagents.',
    'Can you tell me the potential product of a chemical reaction that uses <INPUT> as the reactants and reagents?',
    'Using <INPUT> as the reactants and reagents, tell me the potential product.',
    'Predict a possible product from the listed reactants and reagents. <INPUT>',
    '<INPUT> Considering the given starting materials, what might be the resulting product in a chemical reaction?',
    'A chemical reaction has started with the substance(s) <INPUT> as the reactants and reagents, what could be a probable product?'
]

RS = [
    'Based on the given product, provide some plausible reactants that might have been utilized to prepare it. <INPUT>',
    'Can you identify the reactant(s) that might result in the given product <INPUT> ?',
    'Given the following product, please provide possible reactants. <INPUT>',
    'Do retrosynthesis with the product <INPUT> .',
    '<INPUT> Given the product provided, propose some possible reactants that could have been employed in its formation.',
    'To synthesis <INPUT>, what are the possible reactants? Write in the SELFIES representation.',
    'Provide the potential reactants that may be used to produce the product <INPUT> .',
    'What reactants could lead to the production of the following product? <INPUT>',
    'With the given product <INPUT>, suggest some likely reactants that were used in its synthesis.',
    'Identify possible reactants that could have been used to create the specified product. <INPUT>',
    'Could you tell which reactants might have been used to generate the following product? <INPUT>',
    'Suggest possible substances that may have been involved in the synthesis of the presented compound. <INPUT>',
    'Can you list the reactants that might result in the chemical product <INPUT> ?'
]

REAGENT = [
    'Based on the given chemical reaction, provide some plausible reagents that might have been utilized to prepare it. <INPUT>',
    'Can you identify the reagents that might result in the given chemical reaction <INPUT> ?',
    'Given the following chemical reaction, please provide possible reagents. <INPUT>',
    '<INPUT> Given the chemical reaction provided, propose some possible reagents that could have been employed in its formation.',
    'With the given chemical reaction <INPUT>, suggest some likely reagents that were used in its synthesis.',
    'Can you provide potential reagents for the following chemical reaction <INPUT> ?',
    'Please suggest some possible reagents that could have been used in the following chemical reaction <INPUT>',
    '<INPUT> Given this chemical reaction, what are some reagents that could have been used?',
    'Can you suggest some reagents that might have been used in the given chemical reaction <INPUT> ?',
    'Given the following chemical reaction, what are some potential reagents that could have been employed? <INPUT>',
]

NC_S2F = [
    '<INPUT> is the SELFIES representation of a molecule. What is its molecular formula?',
    'Convert the SELFIES representation of a molecule <INPUT> into molecular formula.',
    'What is the formula of the molecule <INPUT> ?',
    'Can you give the molecular molecular formula of <INPUT> ?',
    'Please write the molecular formula of the molecule <INPUT> .',
    'Given the SELFIES representation <INPUT>, what would be its molecular formula?',
    'The SELFIES representation <INPUT> represents a specific molecule. Can you reveal its molecular formula?',
    'Considering the SELFIES code <INPUT>, can you determine the corresponding molecular formula?',
    'Can you tell me the molecular formula of <INPUT> ?', "I'd like to know the molecular formula of <INPUT> . Can you tell me?",
    'What is the molecular formula for the molecule denoted by <INPUT> ?',
    'What is the molecular formula of <INPUT> ?',
    'Please provide the molecular formula for <INPUT> .'
]

NC_S2I = [
    '<INPUT> is the SELFIES representation of a molecule. What is its IUPAC name?',
    'Convert the SELFIES representation of a molecule <INPUT> into IUPAC name.',
    'What is the IUPAC name of the molecule <INPUT> ?',
    'Can you give the IUPAC name of the molecule <INPUT> ?',
    'Please write the IUPAC name of the molecule <INPUT> .',
    '<INPUT> The above is a SELFIES representation. Write the IUPAC name of the corresponding molecule.',
    'Determine the IUPAC name for the molecule represented by the following SELFIES representation: <INPUT> .',
    'What is the IUPAC name for the molecule whose SELFIES representation is <INPUT> ?',
    'Determine the IUPAC name for the molecule denoted by <INPUT> .',
    'Translate the given SELFIES formula of a molecule <INPUT> into its IUPAC name.',
    'Provide the IUPAC name for the molecule represented as <INPUT> .',
    'Convert the following SELFIES notation <INPUT> into its IUPAC nomenclature.',
    'Turn the given SELFIES symbol of a molecule <INPUT> into its respective IUPAC name.'
]

NC_I2S = [
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

NC_I2F = [
    '<INPUT> is the IUPAC name of a molecule. Please give its molecular formula.',
    'Convert the IUPAC name of a molecule <INPUT> into molecular formula.',
    'What is the molecular formula of the molecule <INPUT> ?',
    'Can you give the molecular formula of the molecule <INPUT> ?',
    'Please write the molecular formula of the molecule <INPUT> .',
    '<INPUT> The above is the IUPAC name of a molecule. Write its molecular formula.',
    'What is the molecular formula of the molecule identified by <INPUT> ?',
    'For the molecule named <INPUT> by IUPAC nomenclature, provide its molecular formula.',
    'Name the molecular formula for the molecule established as <INPUT> using the IUPAC nomenclature.',
    'Under IUPAC nomenclature, a molecule named <INPUT> is given. What is its corresponding molecular formula?',
    'Translate the chemical IUPAC name <INPUT> into its molecular formula.',
    'Provide the molecular formula for the IUPAC named substance <INPUT> .',
    'What is the molecular formula of the compound with this IUPAC name <INPUT> ?'
]

MC = [
    'Could you give me a brief introduction to this molecule? <INPUT>',
    'Describe this molecule: <INPUT>',
    'Please give me some details about this molecule. <INPUT>',
    'Please provide a brief introduction to this molecule. <INPUT>',
    'Tell me something about this molecule: <INPUT>',
    '<INPUT> The above is a compound. Could you please tell me something about it?',
    '<INPUT> What do you know about the molecule?',
    'Here is a molecule represented with SMILE: <INPUT> . Please describe it in natural language.',
    'Can you briefly describe the molecular encoded by this SMILES notation? <INPUT>',
    'I need a brief explanation of the molecule denoted in this SMILES notation. <INPUT>', "I'd like a short overview about this molecule. Can you do that? <INPUT>",
    'May I have a capsulized explanation for this molecule? <INPUT>',
    'What can you tell me about this molecule? <INPUT>'
]

MG = [
    'Based on the given information, generate a molecule that meets the desired specifications: <INPUT>',
    'Give me a molecule that satisfies the conditions outlined in the description: <INPUT>',
    'Generate a molecule based on this description: <INPUT>',
    'Can you create a molecule that matches the given characteristics? <INPUT>',
    'I need a molecule that meets the following conditions: <INPUT> Please represent the molecule in SELFIES.',
    'Suppose there is a molecule that meets the following description: <INPUT> Please write the SELFIES representation of it.',
    '<INPUT> Use the above information to create a molecule.',
    'Build a molecule that meets the requirement: <INPUT>',
    'Generate a molecule that fulfills the requirement: <INPUT>',
    'Conceptualize a molecule that meets the specified attribute(s): <INPUT>',
    'Come up with a molecule based on the description: <INPUT>',
    'Could you please return a molecule that adheres to this description? <INPUT>',
    'I give you a description of a molecule, and you need to return one molecule in SELFIES that meets the description. The description: <INPUT>'
]

HOMO = [
    "I would like to know the highest occupied molecular orbital (HOMO) energy of <INPUT> , could you please provide it?",
    "Please provide the HOMO energy value for following molecule. <INPUT>",
    "I am interested in the HOMO energy of <INPUT> , could you tell me what it is?",
    "What is the highest occupied molecular orbital (HOMO) energy of <INPUT> ?",
    "Could you give me the HOMO energy value of <INPUT> ?",
    "What is the HOMO energy of <INPUT> ?",
    "Please provide the highest occupied molecular orbital (HOMO) energy value for this molecule.",
    "Please provide me with the HOMO energy value of <INPUT> .",
    "What is the HOMO level of energy for <INPUT> ?",
    "I would like to know the HOMO energy of <INPUT>, could you please provide it?",
    "Can you tell me the value of the HOMO energy for following molecule? <INPUT>",
    "Please provide the highest occupied molecular orbital (HOMO) energy of <INPUT> .",
]

LUMO = [
    "Please provide me with the LUMO energy value of <INPUT> .",
    "I am interested in the LUMO energy of <INPUT> , could you tell me what it is?",
    "I would like to know the lowest unoccupied molecular orbital (LUMO) energy of <INPUT> , could you please provide it?",
    "What is the LUMO energy of <INPUT> ?",
    "What is the LUMO level of energy for following molecule? <INPUT>",
    "I would like to know the LUMO energy of <INPUT> , could you please provide it?",
    "What is the lowest unoccupied molecular orbital (LUMO) energy of <INPUT> ?",
    "Could you give me the LUMO energy value of <INPUT> ?",
    "Please provide the lowest unoccupied molecular orbital (LUMO) energy value for following molecule. <INPUT>",
    "Please provide the lowest unoccupied molecular orbital (LUMO) energy of <INPUT> .",
    "Can you tell me the value of the LUMO energy for <INPUT> ?",
    "Please provide the LUMO energy value for <INPUT> .",
]

HOMO_LUMO_GAP = [
    "Please provide the gap between HOMO and LUMO of <INPUT> .",
    "I would like to know the HOMO-LUMO gap of <INPUT> , can you provide it?",
    "Please give me the HOMO-LUMO gap energy for following molecule. <INPUT>",
    "Can you give me the energy difference between the HOMO and LUMO orbitals of <INPUT> ?",
    "Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of <INPUT> .",
    "I need to know the HOMO-LUMO gap energy of <INPUT> , could you please provide it?",
    "What is the energy separation between the HOMO and LUMO of <INPUT> ?",
    "Could you tell me the energy difference between HOMO and LUMO for <INPUT> ?",
    "What is the HOMO-LUMO gap of <INPUT> ?",
]

BACE = [
    "Does <INPUT> inhibit the BACE-1 enzyme?",
    "Can <INPUT> act as an inhibitor of BACE-1?",
    "Is <INPUT> active against BACE-1?",
    "Could <INPUT> exhibit inhibitory activity towards BACE-1?",
    "Does <INPUT> have the potential to inhibit BACE-1?",
    "Is <INPUT> capable of inhibiting the BACE-1 enzyme?",
    "Would <INPUT> show activity against BACE-1?",
    "Can <INPUT> function as a BACE-1 inhibitor?",
    "Is it likely for <INPUT> to inhibit BACE-1?",
    "Does <INPUT> demonstrate inhibitory effects on BACE-1?",
    "Is <INPUT> a potential BACE-1 inhibitory compound?",
    "Could <INPUT> effectively inhibit the activity of BACE-1?",
    "Does <INPUT> possess properties to inhibit BACE-1?"
]

BBBP = [
    'Can <INPUT> pass through the blood-brain barrier (BBB)?',
    'Can <INPUT> permeate the blood-brain barrier?',
    'Does <INPUT> have the blood-brain barrier permeability (BBBP)?',
    'Does <INPUT> have the ability to penetrate the blood-brain barrier?',
    'Is <INPUT> capable of crossing the blood-brain barrier (BBB)?',
    'Can <INPUT> penetrate the blood-brain barrier?',
    'Is <INPUT> capable of crossing the blood-brain barrier (BBB)?',
    'Is it likely for <INPUT> to effectively traverse the blood-brain barrier?',
    'Is blood-brain barrier permeability (BBBP) present in <INPUT> ?',
    'Does <INPUT> demonstrate the potential for blood-brain barrier permeation (BBBP)?',
    'Does blood-brain barrier permeability (BBBP) apply to <INPUT> ?',
    'Is blood-brain barrier permeability (BBBP) a property of <INPUT> ?',
    'Would <INPUT> be able to successfully permeate the blood-brain barrier?'
]

CLINTOX_FDA = [
    "Can <INPUT> gain approval from the FDA?",
    "Is <INPUT> capable of receiving FDA approval?",
    "Does <INPUT> have the potential to be approved by the FDA?",
    "Is it likely for <INPUT> to be approved by the FDA?",
    "Would <INPUT> be able to successfully obtain FDA approval?",
    "Can <INPUT> meet the criteria for FDA approval?",
    "Is FDA approval applicable to <INPUT> ?",
    "Is FDA approval present for <INPUT> ?",
    "Does FDA approval apply to <INPUT> ?",
    "Is FDA approval a property of <INPUT> ?",
    "Does <INPUT> demonstrate the potential for FDA approval?",
    "Is <INPUT> currently approved by the FDA?",
    "Has <INPUT> been granted FDA approval?"
]

CLINTOX_CT = [
    "Has <INPUT> successfully passed clinical trials?",
    "Did <INPUT> pass its clinical trials?",
    "Is <INPUT> a drug that has passed clinical trials?",
    "Can you confirm if <INPUT> has passed clinical trials?",
    "Does <INPUT> have successful clinical trial results?",
    "Is it true that <INPUT> passed clinical trials?",
    "Has <INPUT> successfully completed clinical trials?",
    "Did <INPUT> achieve positive outcomes in clinical trials?",
    "Is <INPUT> associated with successful clinical trial completion?",
    "Has <INPUT> been proven effective in clinical trials?",
    "Is <INPUT> a drug that successfully passed through clinical testing?",
    "Can <INPUT> be considered to have passed clinical trials?",
    "Has <INPUT> demonstrated success in clinical trials?"
]

TOXCAST = [
    "Does <INPUT> exhibit toxicity in the ACEA_T47D_80hr_Negative assay?",
    "Is <INPUT> toxic according to the ACEA_T47D_80hr_Negative test results?",
    "Can <INPUT> induce a negative response in the ACEA_T47D_80hr_Negative assay?",
    "Does <INPUT> show adverse effects in the ACEA_T47D_80hr_Negative toxicity test?",
    "Is <INPUT> predicted to be non-toxic in the ACEA_T47D_80hr_Negative assay?",
    "Would <INPUT> result in a negative outcome regarding toxicity in the ACEA_T47D_80hr_Negative assay?",
    "Is toxicity associated with <INPUT> in the context of the ACEA_T47D_80hr_Negative assay?",
    "Does <INPUT> demonstrate toxic properties when evaluated by the ACEA_T47D_80hr_Negative test?",
    "Can <INPUT> be considered safe based on the ACEA_T47D_80hr_Negative assay results?",
    "Is it likely for <INPUT> to be toxic in the ACEA_T47D_80hr_Negative toxicity assessment?",
    "Does <INPUT> have the potential to cause toxicity according to the ACEA_T47D_80hr_Negative assay findings?",
    "Could <INPUT> exhibit harmful effects in the ACEA_T47D_80hr_Negative toxicity test?",
    "Is toxicity a characteristic of <INPUT> as determined by the ACEA_T47D_80hr_Negative assay?"
]

TOX21 = [
    "Can <INPUT> cause toxicity via nuclear receptor signaling pathways?",
    "Does <INPUT> exhibit toxic effects in nuclear receptor bioassays?",
    "Is <INPUT> capable of inducing toxicity through nuclear receptor interactions?",
    "Does <INPUT> have toxic potential based on nuclear receptor signaling results?",
    "Can <INPUT> activate nuclear receptors leading to toxic outcomes?",
    "Is <INPUT> associated with toxicity in nuclear receptor signaling assays?",
    "Does <INPUT> demonstrate toxicity mediated by nuclear receptor pathways?",
    "Is it likely for <INPUT> to be toxic according to nuclear receptor bioassay data?",
    "Does nuclear receptor signaling suggest toxicity for <INPUT>?",
    "Is toxicity a property of <INPUT> as indicated by nuclear receptor assays?",
    "Does <INPUT> show potential for toxicity through nuclear receptor modulation?",
    "Could <INPUT> be toxic based on its effects on nuclear receptor signaling?",
    "Is <INPUT> toxic when evaluated in nuclear receptor signaling bioassays?"
]

HIV = [
    'Is <INPUT> known to inhibit HIV replication?',
    'Does <INPUT> inhibit viral replication for HIV?',
    'Could <INPUT> be used to prevent HIV replication?',
    'Can <INPUT> inhibit the replication of human immunodeficiency virus (HIV)?',
    'Could HIV replication be slowed or stopped by <INPUT> ?',
    'Does <INPUT> have an inhibitory impact on HIV?',
    'Can <INPUT> effectively inhibit HIV replication?',
    'Does <INPUT> exhibit inhibitory effects on HIV replication?',
    'Is <INPUT> capable of suppressing HIV replication?',
    'Would <INPUT> have the ability to hinder HIV replication?',
    'Can <INPUT> serve as an inhibitor of HIV replication?',
    'Do you suggest that <INPUT> can impede the replication of HIV?',
    '<INPUT> Predict if the molecule given above have an inhibitory impact on HIV.'
]

QM9_MU = [
    "I would like to know the dipole moment of <INPUT> , could you please provide it?",
    "Please provide the dipole moment value for <INPUT> .",
    "I am interested in the dipole moment of <INPUT> , could you tell me what it is?",
    "What is the dipole moment of <INPUT> ?",
    "Could you give me the dipole moment value of <INPUT> ?",
    "What is the value of the <INPUT> for this molecule?",
    "Please provide the dipole moment value for <INPUT> .",
    "Please provide me with the dipole moment of <INPUT> .",
    "What is the dipole moment measurement for <INPUT> ?",
    "I would like to know the dipole moment of <INPUT> , could you please provide it?",
    "Can you tell me the value of the dipole moment for <INPUT> ?",
    "Please provide the dipole moment of <INPUT> .",
]

QM9_ALPHA = [
    "I would like to know the isotropic polarizability of <INPUT> , could you please provide it?",
    "Please provide the isotropic polarizability value for <INPUT> .",
    "I am interested in the isotropic polarizability of <INPUT> , could you tell me what it is?",
    "What is the isotropic polarizability of <INPUT> ?",
    "Could you give me the isotropic polarizability value of <INPUT> ?",
    "What is the isotropic polarizability of <INPUT> ?",
    "Please provide the isotropic polarizability value for <INPUT> .",
    "Please provide me with the isotropic polarizability of <INPUT> .",
    "What is the isotropic polarizability measurement for <INPUT> ?",
    "I would like to know the isotropic polarizability of <INPUT> , could you please provide it?",
    "Can you tell me the value of the isotropic polarizability for <INPUT> ?",
    "Please provide the isotropic polarizability of <INPUT> .",
]

QM9_R2 = [
    "I would like to know the electronic spatial extent of <INPUT> , could you please provide it?",
    "Please provide the electronic spatial extent value for <INPUT> .",
    "I am interested in the electronic spatial extent of <INPUT> , could you tell me what it is?",
    "What is the electronic spatial extent of <INPUT> ?",
    "Could you give me the electronic spatial extent value of <INPUT> ?",
    "What is the electronic spatial extent of <INPUT> ?",
    "Please provide the electronic spatial extent value for <INPUT> .",
    "Please provide me with the electronic spatial extent of <INPUT> .",
    "What is the electronic spatial extent measurement for <INPUT> ?",
    "I would like to know the electronic spatial extent of <INPUT> , could you please provide it?",
    "Can you tell me the value of the electronic spatial extent for <INPUT> ?",
    "Please provide the electronic spatial extent of <INPUT> .",
]

QM9_ZPVE = [
    "I would like to know the zero point vibrational energy of <INPUT> , could you please provide it?",
    "Please provide the zero point vibrational energy value for <INPUT> .",
    "I am interested in the zero point vibrational energy of <INPUT> , could you tell me what it is?",
    "What is the zero point vibrational energy of <INPUT> ?",
    "Could you give me the zero point vibrational energy value of <INPUT> ?",
    "What is the zero point vibrational energy of <INPUT> ?",
    "Please provide the zero point vibrational energy value for <INPUT> .",
    "Please provide me with the zero point vibrational energy of <INPUT> .",
    "What is the zero point vibrational energy measurement for <INPUT> ?",
    "I would like to know the zero point vibrational energy of <INPUT> , could you please provide it?",
    "Can you tell me the value of the zero point vibrational energy for <INPUT> ?",
    "Please provide the zero point vibrational energy of <INPUT> .",
]

QM9_CV = [
    "I would like to know the heat capacity of <INPUT> at 298.15K, could you please provide it?",
    "Please provide the heat capacity value at 298.15K for <INPUT> .",
    "I am interested in the heat capacity of <INPUT> at 298.15K, could you tell me what it is?",
    "What is the heat capacity of <INPUT> at 298.15K?",
    "Could you give me the heat capacity value of <INPUT> at 298.15K?",
    "What is the heat capacity at 298.15K of <INPUT> ?",
    "Please provide the heat capacity value at 298.15K for <INPUT> .",
    "Please provide me with the heat capacity of <INPUT> at 298.15K.",
    "What is the heat capacity measurement at 298.15K for <INPUT> ?",
    "I would like to know the heat capacity at 298.15K of <INPUT> , could you please provide it?",
    "Can you tell me the value of the heat capacity for <INPUT> at 298.15K?",
    "Please provide the heat capacity of <INPUT> at 298.15K.",
]

QM9_U298 = [
    "I would like to know the internal energy of <INPUT> at 298.15K, could you please provide it?",
    "Please provide the internal energy value at 298.15K for <INPUT> .",
    "I am interested in the internal energy of <INPUT> at 298.15K, could you tell me what it is?",
    "What is the internal energy of <INPUT> at 298.15K?",
    "Could you give me the internal energy value of <INPUT> at 298.15K?",
    "What is the internal energy at 298.15K of <INPUT> ?",
    "Please provide the internal energy value at 298.15K for <INPUT> .",
    "Please provide me with the internal energy of <INPUT> at 298.15K.",
    "What is the internal energy measurement at 298.15K for <INPUT> ?",
    "I would like to know the internal energy at 298.15K of <INPUT> , could you please provide it?",
    "Can you tell me the value of the internal energy for <INPUT> at 298.15K?",
    "Please provide the internal energy of <INPUT> at 298.15K.",
]

QM9_H298 = [
    "I would like to know the enthalpy of <INPUT> at 298.15K, could you please provide it?",
    "Please provide the enthalpy value at 298.15K for <INPUT> .",
    "I am interested in the enthalpy of <INPUT> at 298.15K, could you tell me what it is?",
    "What is the enthalpy of <INPUT> at 298.15K?",
    "Could you give me the enthalpy value of <INPUT> at 298.15K?",
    "What is the enthalpy at 298.15K of <INPUT> ?",
    "Please provide the enthalpy value at 298.15K for <INPUT> .",
    "Please provide me with the enthalpy of <INPUT> at 298.15K.",
    "What is the enthalpy measurement at 298.15K for <INPUT> ?",
    "I would like to know the enthalpy at 298.15K of <INPUT> , could you please provide it?",
    "Can you tell me the value of the enthalpy for <INPUT> at 298.15K?",
    "Please provide the enthalpy of <INPUT> at 298.15K.",
]

QM9_G298 = [
    "I would like to know the free energy of <INPUT> at 298.15K, could you please provide it?",
    "Please provide the free energy value at 298.15K for <INPUT> .",
    "I am interested in the free energy of <INPUT> at 298.15K, could you tell me what it is?",
    "What is the free energy of <INPUT> at 298.15K?",
    "Could you give me the free energy value of <INPUT> at 298.15K?",
    "What is the free energy at 298.15K of <INPUT> ?",
    "Please provide the free energy value at 298.15K for <INPUT> .",
    "Please provide me with the free energy of <INPUT> at 298.15K.",
    "What is the free energy measurement at 298.15K for <INPUT> ?",
    "I would like to know the free energy at 298.15K of <INPUT> , could you please provide it?",
    "Can you tell me the value of the free energy for <INPUT> at 298.15K?",
    "Please provide the free energy of <INPUT> at 298.15K.",
]

ESOL = [
    'What is the log solubility of <INPUT> in water?',
    'Please predict the log solubility of <INPUT> in water.',
    'Tell me the solubility of <INPUT> in water.',
    'How soluble is <INPUT> ?',
    'What concentration of <INPUT> can be dissolved in water?',
    'What is the solubility of <INPUT> ?',
    'How many mols of <INPUT> can be dissolved in a liter of water?',
    'Could you provide the logarithmic solubility value of <INPUT> in aqueous solutions?',
    'Give me the log water solubility for the following molecule: <INPUT> .',
    '<INPUT> What is the log solubility in water for the molecule given above?',
    'What is the logarithmic value of the solubility of <INPUT> in water??',
    'Can you predict the water solubility of <INPUT> ?',
    '<INPUT> Could you tell me the log solubility in water for the molecule given above?'
]

LIPO = [
    'What is the octanol/water distribution coefficient (logD at pH 7.4) of <INPUT> ?',
    'Please predict the octanol/water distribution coefficient (logD at pH 7.4) of <INPUT> .',
    'Tell me the octanol/water distribution coefficient (logD at pH 7.4) of <INPUT> .',
    'Predict the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for <INPUT> .',
    '<INPUT> What is the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for the molecule given above?',
    'For this molecule <INPUT>, predict the octanol/water distribution coefficient (logD at pH 7.4).',
    'Could you provide information on the octanol/water distribution coefficient (logD) of <INPUT> under pH 7.4?',
    'At pH 7.4, what is the logD value (the octanol/water distribution coefficient) of <INPUT> ?',
    '<INPUT> What is the logD (the octanol/water distribution coefficient) at pH 7.4 for the given molecule?',
    'What value does the octanol/water distribution coefficient (logD at pH 7.4) have for <INPUT> ?',
    'What is the logD at pH 7.4 (the octanol/water distribution coefficient) associated with <INPUT> ?',
    'Provide the logD (the octanol/water distribution coefficient) at pH 7.4 of <INPUT> .',
    '<INPUT> Please predict the octanol/water distribution coefficient (logD at pH 7.4) for the molecule given above.'
]