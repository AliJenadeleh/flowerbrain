import Brain.brain
def main():
        print('main called')
        b = Brain.brain.brain()
        b.training()
        b.SuggestUnknow()
        #b.ShowDataSuggestion()

        

if __name__ == '__main__':
        main()