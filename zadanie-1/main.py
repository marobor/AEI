from clp3 import clp

swap_tuple = (
        ('a', 'ą'),
        ('c', 'ć'),
        ('e', 'ę'),
        ('l', 'ł'),
        ('o', 'ó'),
        ('s', 'ś'),
        ('z', 'ż', 'ź')
    )


def find_tuple(letter):
    for i, current_tuple in enumerate(swap_tuple):
        if letter in current_tuple:
            return i
    return None


def word_combinations(word):
    combinations_list = []

    if clp(word):
        return ''

    def generate_combinations(current_word, index):
        nonlocal combinations_list

        if index == len(current_word):
            combinations_list.append(current_word)
            return

        current_char = current_word[index]
        tuple_index = find_tuple(current_char)

        if tuple_index is not None:
            replacements = swap_tuple[tuple_index]
            for replacement in replacements:
                new_word = current_word[:index] + replacement + current_word[index + 1:]
                generate_combinations(new_word, index + 1)
        else:
            generate_combinations(current_word, index + 1)

    generate_combinations(word, 0)

    return combinations_list


result_1 = word_combinations('kot')
result_2 = word_combinations('aóz')
result_3 = word_combinations('trzesawisko')
print(result_1, '\n')
print(result_2, '\n')
print(result_3, '\n')