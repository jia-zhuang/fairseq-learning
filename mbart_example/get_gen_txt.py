import argparse

def get_ref_and_hyp(fname):
    ref_list, hyp_list = [], []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith('T-'):  # ref
                ref_list.append(get_idx_line(line))
            elif line.startswith('H-'):  # hyp
                hyp_list.append(get_idx_line(line))
    
    assert len(ref_list) == len(hyp_list)

    ref_list = sorted(ref_list)
    hyp_list = sorted(hyp_list)

    return ref_list, hyp_list


def get_idx_line(line):
    parts = line.split('\t')
    idx = int(parts[0].split('-')[-1])
    txt = parts[-1]
    return idx, txt


def write_to_file(ref_list, hyp_list, prefix=''):
    with open(prefix + 'T.txt', 'w') as f_ref, open(prefix + 'H.txt', 'w') as f_hyp:
        for ref, hyp in zip(ref_list, hyp_list):
            f_ref.write(ref[1] + '\n')
            f_hyp.write(hyp[1] + '\n')


def main(idx):
    fname = f'output{idx}.txt'
    ref_list, hyp_list = get_ref_and_hyp(fname)
    write_to_file(ref_list, hyp_list, prefix=f'epoch{idx}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str)
    args = parser.parse_args()
    main(args.idx)
