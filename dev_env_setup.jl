# reference URL. https://qiita.com/mametank/items/43330a9452f0039ca22d

import Pkg
Pkg.activate("../ADFMCOMMS2.jl")
Pkg.activate(".")
Pkg.develop(path="../ADFMCOMMS2.jl")
Pkg.instantiate()


using Revise
import ADFMCOMMS2
import SignalFlow
