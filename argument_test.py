import argparse

def main(args):
    if args.name:
        print(int(args.number)*100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',action = 'store_true')
    parser.add_argument('--number',default=None)
    #parser.add_argument('--model', required=True)
    args = parser.parse_args()
    
    main(args)
