// torch/extention.h imports torch/python.h, which completely breaks intellisense, so we have to manually exclude it while editing the code.
// This file serves as a sort of macro that does this automatically by simply including it.
#ifndef EDITMODE
    #ifndef TORCH_INDUCTOR_CPP_WRAPPER
        #include <torch/extension.h>
    #endif
#endif
#ifdef EDITMODE
    #include <vector>
    #include <string>
    #include <torch/all.h>
#endif