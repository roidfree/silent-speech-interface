# Electrode Comparison Summary

## Operational
- agagcl: test accuracy=0.9556, macro_f1=0.4851, channels=1,2,3,5,6,7,8
- pedot: test accuracy=1.0000, macro_f1=1.0000, channels=2,3,4,5,7,8

## Matched
- matched_channels=2,3,4,5,7,8
- agagcl: no held-out split available
- pedot: no held-out split available

## Caveats
- agagcl: 55/57 recordings passed base QC.
- pedot: 10/11 recordings passed base QC. 1 recordings include suspicious channels.

## Matched Trial Budget
- train bye/s1: budget=1, agagcl=45, pedot=1
- train help/s1: budget=1, agagcl=44, pedot=1
- train no/s1: budget=1, agagcl=29, pedot=1
- train wait/s1: budget=1, agagcl=22, pedot=1