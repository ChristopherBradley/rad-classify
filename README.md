# rad-classify
Practicing text classification techniques

## Setup Instructions

0. Clone the repository and navigate to it with: 
`git clone https://github.com/ChristopherBradley/rad-classify.git`
`cd rad-classify`

1. Download miniconda from [here](https://docs.conda.io/en/latest/miniconda.html#macos-installers)
Create an environment with `conda create --name rad-classify python=3.8`  
Activate the environment with `conda activate rad-classify`

2. Install dependencies with: 
`pip install --editable .`
`pip install numpy pandas sklearn fasttext`

1. Download the WOS dataset from https://data.mendeley.com/datasets/9rw3vkcfy4/2  
and place in the data directory so it looks like data/WebOfScience/...

