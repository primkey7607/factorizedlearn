from sys import maxsize

class RequestCache:
  def __init__(self, cache_size):
    self.cache_size = cache_size
    self.cnt = 0
    self.cdict = {}
  
  def find_lfu(self):
    min_freq = maxsize
    cur_sc = None
    for sc in self.cdict:
      aug_plan, freq_cnt = self.cdict[sc]
      if freq_cnt < min_freq:
        cur_sc = sc
        min_freq = freq_cnt
    
    return (cur_sc, min_freq)
    

  def read_el(self, schema : tuple):
    if schema in self.cdict:
      aug_plan, freq_cnt = self.cdict[schema]
      #update freq count
      self.cdict[schema] = (aug_plan, freq_cnt + 1)
      return aug_plan
    else:
      #now, we don't have a match.
      #if we have more items than entries in the dictionary...
      if self.cnt >= self.cache_size:
        #evict LFU
        min_sc, f_cnt = self.find_lfu()
        del self.cdict[min_sc]
        self.cnt -= 1
      #add schema and aug plan...is what we'd like to do, but we don't have the aug plan when we check this.
      #so, return None at this point.
      return None
    
    #...and then, later, when we have the aug plan, we can add it.
    #the aug plan is specified as a dictionary, where key is the join key to a seller table,
    #and the value is the set of attributes to use
    def add_el(self, schema : tuple, aug_plan : list):
      if self.cnt >= self.cache_size:
        raise Exception("add_el on full cache")
      
      self.cdict[schema] = (aug_plan, 1)
      self.cnt += 1

