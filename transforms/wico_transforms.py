class WICOTransforms:
    
    def wico_5g_vs_non_conspiracy(wico):
        return [g for g in wico if g.y != 2]
    