from biblib.bib import Parser
import jellyfish
from tqdm import tqdm


def small_gauss(n):
    return n * (n + 1) // 2


def get_entries(*paths):
    entries = []
    for path in paths:
        with open(path, 'r') as fp:
            string = []
            running = True
            level = 0
            while running:
                char = fp.read(1)
                if char == "@" and level == 0:
                    entry = Parser().parse(''.join(string)).get_entries()
                    entries.extend(list(entry.items()))
                    string = [char]
                elif char == "":
                    entry = Parser().parse(''.join(string)).get_entries()
                    entries.extend(list(entry.items()))
                    running = False
                else:
                    if char == "{":
                        level += 1
                    elif char == "}":
                        level -= 1
                    string.append(char)
    return entries


def find_duplicate_keys(*paths):
    entries = get_entries(*paths)
    keys = [pt[0] for pt in entries]
    exclude = []
    for key in keys:
        if key in exclude:
            continue
        if keys.count(key) > 1:
            exclude.append(key)
    return exclude


def find_duplicate_titles(*paths):
    no_title = []
    entries = get_entries(*paths)
    titles = []
    keys = []
    excludes = []
    for ent in entries:
        key, entry = ent
        if 'title' in entry:
            titles.append(entry['title'])
            keys.append(key)
        else:
            no_title.append(key)

    for title in titles:
        if title in excludes:
            continue
        if titles.count(title) > 1:
            excludes.append(title)
    return excludes, no_title


def find_close_titles(*paths, threshold=5, verbose=False):
    entries = get_entries(*paths)
    titles = []
    for ent in entries:
        key, entry = ent
        if 'title' in entry:
            titles.append(entry['title'])
    
    ret = {}
    with tqdm(total=small_gauss(len(titles)) - len(titles),
              disable=not verbose) as pbar:
        for i, title in enumerate(titles):
            if title in ret:
                ret[title].append(title)
            else:
                ret[title] = []
            if i == len(titles) - 1:
                break
            for otitle in titles[i+1:]:
                dist = min(jellyfish.hamming_distance(title, otitle),
                           jellyfish.levenshtein_distance(title, otitle))
                if dist <= threshold:
                    ret[title].append(otitle)
                pbar.update(1)
    return ret
