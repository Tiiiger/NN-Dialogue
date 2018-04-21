processed = []
with open("OpenSubData/t_given_s_dialogue_length2_6.txt", "w") as f:
    for line in f.readlines():
        _, t = line.split('l')
        t = t.split()
        if len(t) < 20:
            processed.append(line)
with open("processed/t_given_s_dialogue_length2_6_19.txt", "w") as to:
    to.writelines(processed)
