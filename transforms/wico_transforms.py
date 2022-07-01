class WICOTransforms:
    
    def wico_5g_vs_non_conspiracy(wico):
        return [g for g in wico if g.y != 2]
    
    def wico_5g_vs_non_conspiracy_downsampled_balanced(wico):
        ones = [data for data in wico if data.y == 1]
        zeros = [data for data in wico if data.y == 0]

        return ones + zeros[:412]
    