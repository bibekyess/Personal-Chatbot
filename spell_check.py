import spellchecker


def correct_typos(sentence):
    # Initialize the spell checker object
    spell = spellchecker.SpellChecker(language="en")
    # Adds Bibek to its frequency dictionary to make it a known word
    spell.word_frequency.load_words(
        [
            "Bibek",
            "Bibek's",
            "skillsets",
            "skillset",
            "CV",
            "RIRO",
            "Bisonai",
            "IC",
            "BMC",
            "KAIST",
        ]
    )
    sentence_split = sentence.split()
    # Find the typos in the input sentence
    typos = spell.unknown(sentence_split)
    # Correct the typos
    corrected_sentence = [
        spell.correction(word)
        if spell.correction(word)
        else word
        if word in typos
        else word
        for word in sentence_split
    ]
    # Return the corrected sentence as a string
    return " ".join(corrected_sentence)
