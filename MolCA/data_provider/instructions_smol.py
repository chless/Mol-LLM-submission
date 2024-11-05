forward_reaction_prediction = [
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

retrosynthesis = [
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

reagent_prediction = [
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

smol_name_conversion_s2f = [
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

smol_name_conversion_s2i = [
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

smol_name_conversion_i2s = [
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

smol_molecule_captioning = [
    'Could you give me a brief introduction to this molecule? <INPUT>',
    'Describe this molecule: <INPUT>',
    'Please give me some details about this molecule. <INPUT>',
    'Please provide a brief introduction to this molecule. <INPUT>',
    'Tell me something about this molecule: <INPUT>',
    '<INPUT> The above is a compound. Could you please tell me something about it?',
    '<INPUT> What do you know about the molecule?',
    'Here is a molecule represented with SMILE: <INPUT> . Please describe it in natural language.',
    'Can you briefly describe the molecular encoded by this SELFIES notation? <INPUT>',
    'I need a brief explanation of the molecule denoted in this SELFIES notation. <INPUT>', "I'd like a short overview about this molecule. Can you do that? <INPUT>",
    'May I have a capsulized explanation for this molecule? <INPUT>',
    'What can you tell me about this molecule? <INPUT>'
]

smol_molecule_generation = [
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

homo = [
    "<INPUT> Based on the molecule given above, suggest the HOMO energy.",
    "Based on the given molecule: <INPUT>, what HOMO energy could potentially be measured?",
    "Given the following molecule, please provide the HOMO energy. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable HOMO energy?",
    "Please provide the HOMO energy value for this molecule: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the HOMO energy?",
    "Propose the HOMO energy value given this molecule. <INPUT>",
    "Predict the HOMO energy of the molecule <INPUT>.",
    "Can you tell me the HOMO energy of the molecule that uses <INPUT>?",
    "Using <INPUT> as the molecule, tell me the HOMO energy value.",
    "Predict the possible HOMO energy for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the HOMO energy?",
    "A molecule <INPUT> is given; what could be the HOMO energy?"
]

lumo = [
    "<INPUT> Based on the molecule given above, suggest the LUMO energy.",
    "Based on the given molecule: <INPUT>, what LUMO energy could potentially be measured?",
    "Given the following molecule, please provide the LUMO energy. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable LUMO energy?",
    "Please provide the LUMO energy value for this molecule: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the LUMO energy?",
    "Propose the LUMO energy value given this molecule. <INPUT>",
    "Predict the LUMO energy of the molecule <INPUT>.",
    "Can you tell me the LUMO energy of the molecule that uses <INPUT>?",
    "Using <INPUT> as the molecule, tell me the LUMO energy value.",
    "Predict the possible LUMO energy for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the LUMO energy?",
    "A molecule <INPUT> is given; what could be the LUMO energy?"
]

homo_lumo_gap = [
    "<INPUT> Based on the molecule given above, suggest the HOMO-LUMO gap.",
    "Based on the given molecule: <INPUT>, what HOMO-LUMO gap could potentially be measured?",
    "Given the following molecule, please provide the HOMO-LUMO gap. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable HOMO-LUMO gap?",
    "Please provide the HOMO-LUMO gap value for this molecule: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the HOMO-LUMO gap?",
    "Propose the HOMO-LUMO gap value given this molecule. <INPUT>",
    "Predict the HOMO-LUMO gap of the molecule <INPUT>.",
    "Can you tell me the HOMO-LUMO gap of the molecule that uses <INPUT>?",
    "Using <INPUT> as the molecule, tell me the HOMO-LUMO gap value.",
    "Predict the possible HOMO-LUMO gap for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the HOMO-LUMO gap?",
    "A molecule <INPUT> is given; what could be the HOMO-LUMO gap?"
]

qm9_dipole_moment = [
"<INPUT> Based on the molecule given above, could you please provide its dipole moment?"
"Based on the given molecule: <INPUT>, could you please provide its dipole moment?"
"Given the following molecule, please provide its dipole moment. <INPUT>"
"<INPUT> Given the above molecule, could you please provide its dipole moment?"
"Please provide the dipole moment that could be measured for this molecule: <INPUT>."
"Consider that for a molecule, if <INPUT> is given, what is its dipole moment?"
"Propose the dipole moment for this molecule. <INPUT>"
"Predict the dipole moment of a molecule with <INPUT>."
"Can you tell me the dipole moment of a molecule that has the structure <INPUT>?"
"Using <INPUT> as the molecule, could you please provide its dipole moment?"
"Predict the dipole moment from the listed molecule. <INPUT>"
"<INPUT> Considering the given molecule, what is its dipole moment?"
"A molecule has the structure <INPUT>, what could be its dipole moment?"
]

qm9_isotropic_polarizability = [
"<INPUT> Based on the molecule given above, could you please provide its isotropic polarizability?",
"Based on the given molecule: <INPUT>, what is its isotropic polarizability?",
"Given the following molecule, please provide its isotropic polarizability. <INPUT>",
"<INPUT> Given the above molecule, what is its isotropic polarizability?",
"Please provide a feasible isotropic polarizability value for this molecule: <INPUT>.",
"Consider that for a molecule, if <INPUT> is given, what is its isotropic polarizability?",
"Propose the isotropic polarizability for this molecule. <INPUT>",
"Predict the isotropic polarizability of a molecule with <INPUT>.",
"Can you tell me the isotropic polarizability of a molecule that has the structure <INPUT>?",
"Using <INPUT> as the molecule, please tell me its isotropic polarizability.",
"Predict the isotropic polarizability from the listed molecule. <INPUT>",
"<INPUT> Considering the given molecule, what is its isotropic polarizability?",
"A molecule has the structure <INPUT>, what could be its isotropic polarizability?"
]

qm9_electronic_spatial_extent = [
"<INPUT> Based on the molecule given above, could you please provide its electronic spatial extent?",
"Based on the given molecule: <INPUT>, what is its electronic spatial extent?",
"Given the following molecule, please provide its electronic spatial extent. <INPUT>",
"<INPUT> Given the above molecule, what is its electronic spatial extent?",
"Please provide the electronic spatial extent that could be measured for this molecule: <INPUT>.",
"Consider that for a molecule, if <INPUT> is given, what is its electronic spatial extent?",
"Propose the electronic spatial extent for this molecule. <INPUT>",
"Predict the electronic spatial extent of a molecule with <INPUT>.",
"Can you tell me the electronic spatial extent of a molecule that has the structure <INPUT>?",
"Using <INPUT> as the molecule, please tell me its electronic spatial extent.",
"Predict the electronic spatial extent from the listed molecule. <INPUT>",
"<INPUT> Considering the given molecule, what is its electronic spatial extent?",
"A molecule has the structure <INPUT>, what could be its electronic spatial extent?"
]

qm9_zero_point_vibrational_energy = [
"<INPUT> Based on the molecule given above, could you please provide its zero point vibrational energy?",
"Based on the given molecule: <INPUT>, what is its zero point vibrational energy?",
"Given the following molecule, please provide its zero point vibrational energy. <INPUT>",
"<INPUT> Given the above molecule, what is its zero point vibrational energy?",
"Please provide a feasible zero point vibrational energy value for this molecule: <INPUT>.",
"Consider that for a molecule, if <INPUT> is given, what is its zero point vibrational energy?",
"Propose the zero point vibrational energy for this molecule. <INPUT>",
"Predict the zero point vibrational energy of a molecule with <INPUT>.",
"Can you tell me the zero point vibrational energy of a molecule that has the structure <INPUT>?",
"Using <INPUT> as the molecule, please tell me its zero point vibrational energy.",
"Predict the zero point vibrational energy from the listed molecule. <INPUT>",
"<INPUT> Considering the given molecule, what is its zero point vibrational energy?",
"A molecule has the structure <INPUT>, what could be its zero point vibrational energy?"
]

qm9_heat_capacity_298K = [
"<INPUT> Based on the molecule given above, could you please provide its heat capacity at 298.15K?"
"Based on the given molecule: <INPUT>, what is its heat capacity at 298.15K?"
"Given the following molecule, please provide its heat capacity at 298.15K. <INPUT>"
"<INPUT> Given the above molecule, what could be its heat capacity at 298.15K?"
"Please provide a feasible heat capacity at 298.15K that could be measured for this molecule: <INPUT>."
"Consider that for a molecule, if <INPUT> is given, what can be its heat capacity at 298.15K?"
"Propose the heat capacity at 298.15K for this molecule. <INPUT>"
"Predict the heat capacity at 298.15K of a molecule with <INPUT>."
"Can you tell me the heat capacity at 298.15K of a molecule that has the structure <INPUT>?"
"Using <INPUT> as the molecule, please tell me its heat capacity at 298.15K."
"Predict the heat capacity at 298.15K from the listed molecule. <INPUT>"
"<INPUT> Considering the given molecule, what might be its heat capacity at 298.15K?"
"A molecule has the structure <INPUT>, what could be its heat capacity at 298.15K?"
]

qm9_internal_energy_298K = [
"<INPUT> Based on the molecule given above, could you please provide its internal energy at 298.15K?"
"Based on the given molecule: <INPUT>, what is its internal energy at 298.15K?"
"Given the following molecule, please provide its internal energy at 298.15K. <INPUT>"
"<INPUT> Given the above molecule, what could be its internal energy at 298.15K?"
"Please provide a feasible internal energy at 298.15K that could be measured for this molecule: <INPUT>."
"Consider that for a molecule, if <INPUT> is given, what can be its internal energy at 298.15K?"
"Propose the internal energy at 298.15K for this molecule. <INPUT>"
"Predict the internal energy at 298.15K of a molecule with <INPUT>."
"Can you tell me the internal energy at 298.15K of a molecule that has the structure <INPUT>?"
"Using <INPUT> as the molecule, please tell me its internal energy at 298.15K."
"Predict the internal energy at 298.15K from the listed molecule. <INPUT>"
"<INPUT> Considering the given molecule, what might be its internal energy at 298.15K?"
"A molecule has the structure <INPUT>, what could be its internal energy at 298.15K?"
]

qm9_enthalpy_298K = [
"<INPUT> Based on the molecule given above, could you please provide its enthalpy at 298.15K?"
"Based on the given molecule: <INPUT>, what is its enthalpy at 298.15K?"
"Given the following molecule, please provide its enthalpy at 298.15K. <INPUT>"
"<INPUT> Given the above molecule, what could be its enthalpy at 298.15K?"
"Please provide a feasible enthalpy at 298.15K that could be measured for this molecule: <INPUT>."
"Consider that for a molecule, if <INPUT> is given, what can be its enthalpy at 298.15K?"
"Propose the enthalpy at 298.15K for this molecule. <INPUT>"
"Predict the enthalpy at 298.15K of a molecule with <INPUT>."
"Can you tell me the enthalpy at 298.15K of a molecule that has the structure <INPUT>?"
"Using <INPUT> as the molecule, please tell me its enthalpy at 298.15K."
"Predict the enthalpy at 298.15K from the listed molecule. <INPUT>"
"<INPUT> Considering the given molecule, what might be its enthalpy at 298.15K?"
"A molecule has the structure <INPUT>, what could be its enthalpy at 298.15K?"
]

qm9_free_energy_298K = [
"<INPUT> Based on the molecule given above, could you please provide its free energy at 298.15K?"
"Based on the given molecule: <INPUT>, what is its free energy at 298.15K?"
"Given the following molecule, please provide its free energy at 298.15K. <INPUT>"
"<INPUT> Given the above molecule, what could be its free energy at 298.15K?"
"Please provide a feasible free energy at 298.15K that could be measured for this molecule: <INPUT>."
"Consider that for a molecule, if <INPUT> is given, what can be its free energy at 298.15K?"
"Propose the free energy at 298.15K for this molecule. <INPUT>"
"Predict the free energy at 298.15K of a molecule with <INPUT>."
"Can you tell me the free energy at 298.15K of a molecule that has the structure <INPUT>?"
"Using <INPUT> as the molecule, please tell me its free energy at 298.15K."
"Predict the free energy at 298.15K from the listed molecule. <INPUT>"
"<INPUT> Considering the given molecule, what might be its free energy at 298.15K?"
"A molecule has the structure <INPUT>, what could be its free energy at 298.15K?"
]


bace = [
    "<INPUT> Based on the molecule given above, suggest the biological activity against BACE-1.",
    "Based on the given molecule: <INPUT>, what biological activity could potentially be observed against BACE-1?",
    "Given the following molecule, please provide the biological activity against BACE-1. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable biological activity against BACE-1?",
    "Please provide the biological activity value for this molecule against BACE-1: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the biological activity against BACE-1?",
    "Propose the biological activity value given this molecule against BACE-1. <INPUT>",
    "Predict the biological activity of the molecule <INPUT> against BACE-1.",
    "Can you tell me the biological activity of the molecule that uses <INPUT> against BACE-1?",
    "Using <INPUT> as the molecule, tell me the biological activity against BACE-1.",
    "Predict the possible biological activity against BACE-1 for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the biological activity against BACE-1?",
    "A molecule <INPUT> is given; what could be the biological activity against BACE-1?"
]

bbbp = [
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

clintox_FDA_APPROVED = [
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

clintox_CT_TOX = [
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

toxcast = [
    "<INPUT> Based on the molecule given above, suggest if it is toxic in the ACEA_T47D_80hr_Negative assay.",
    "Based on the given molecule: <INPUT>, what is the toxicity status according to the ACEA_T47D_80hr_Negative assay?",
    "Given the following molecule, please provide its toxicity information from the ACEA_T47D_80hr_Negative assay. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable toxicity result in the ACEA_T47D_80hr_Negative assay?",
    "Please provide the toxicity status for this molecule based on the ACEA_T47D_80hr_Negative assay: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is its toxicity status according to the ACEA_T47D_80hr_Negative assay?",
    "Propose the toxicity status given this molecule against the ACEA_T47D_80hr_Negative assay. <INPUT>",
    "Predict the toxicity of the molecule <INPUT> in the ACEA_T47D_80hr_Negative assay.",
    "Can you tell me the toxicity status of the molecule that uses <INPUT> in the ACEA_T47D_80hr_Negative assay?",
    "Using <INPUT> as the molecule, tell me the toxicity status in the ACEA_T47D_80hr_Negative assay.",
    "Predict the possible toxicity for the listed molecule in the ACEA_T47D_80hr_Negative assay. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the toxicity result in the ACEA_T47D_80hr_Negative assay?",
    "A molecule <INPUT> is given; what could be the toxicity status in the ACEA_T47D_80hr_Negative assay?"
]

tox21 = [
    "<INPUT> Based on the molecule given above, suggest if it is toxic based on nuclear receptor signaling bioassays.",
    "Based on the given molecule: <INPUT>, what is the toxicity status according to nuclear receptor signaling bioassays?",
    "Given the following molecule, please provide its toxicity information from nuclear receptor signaling bioassays. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable toxicity result in nuclear receptor signaling bioassays?",
    "Please provide the toxicity status for this molecule based on nuclear receptor signaling bioassays: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is its toxicity status according to nuclear receptor signaling bioassays?",
    "Propose the toxicity status given this molecule in nuclear receptor signaling bioassays. <INPUT>",
    "Predict the toxicity of the molecule <INPUT> in nuclear receptor signaling bioassays.",
    "Can you tell me the toxicity status of the molecule that uses <INPUT> in nuclear receptor signaling bioassays?",
    "Using <INPUT> as the molecule, tell me the toxicity status in nuclear receptor signaling bioassays.",
    "Predict the possible toxicity for the listed molecule in nuclear receptor signaling bioassays. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the toxicity result in nuclear receptor signaling bioassays?",
    "A molecule <INPUT> is given; what could be the toxicity status in nuclear receptor signaling bioassays?"
]

hiv = [
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

esol = [
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

lipo = [
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

