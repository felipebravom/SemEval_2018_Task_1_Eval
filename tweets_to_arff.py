# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# tweets_to_arff.py
# felipebravom
# Running example: python tweets_to_arff data/anger-ratings-0to1.test.target.tsv data/anger-ratings-0to1.test.target.arff
# Descrition: Converts SemEval-2018 Task 1 data into arff format.
# usage: python tweets_to_arff <data_type> <input_file> <output file> 
# data_type: 1 for regression (EI and V), 2 for ordinal classification (EI and V), and 3 for multi-label emotion classification

import sys
import os.path
import re

def create_reg_arff(input_file,output_file):
    """
    Creates an arff dataset for the regression data
    """
  

    out=open(output_file,"w")  
    
    f=open(input_file, "rb")
    lines=f.readlines()
    
    in_header=True
    for line in lines:
        if(in_header):           
            header_line='@relation '+os.path.basename(input_file)+'\n\n@attribute ID string \n@attribute Tweet string\n@attribute Affect_Dimension string\n@attribute Intensity_Score numeric \n\n@data\n'
            out.write(header_line)
            in_header=False
        else:
            parts=line.split("\t")
            if len(parts)==4:
         
                id=parts[0]
                tweet=parts[1].replace('\n','')
                tweet=re.sub('\'', '\"', tweet)
                emotion=parts[2]
                score=parts[3].strip() 
                score = score if score != "NONE" else "?"
                
                out_line='\''+id+'\',\''+tweet+'\','+'\''+emotion+'\','+score+'\n'
                out.write(out_line)
            else:
                print "Wrong format"
    

    f.close()  
    out.close()  
    
    
    
def create_oc_arff(input_file,output_file):
    """
    Creates an arff dataset for the ordinal classification data
    """
    out=open(output_file,"w")  
    
    f=open(input_file, "rb")
    lines=f.readlines()
    if (len(lines)>1):
    
        firt_data_line=True
        for line in lines[1:len(lines)]:
            if(firt_data_line):   
                parts=line.split("\t")
                if len(parts)==4:
                    parts=line.split("\t")
                    class_value = "{0,1,2,3}" if parts[2] != "valence" else "{-3,-2,-1,0,1,2,3}"                                
                    header_line='@relation '+os.path.basename(input_file)+'\n\n@attribute ID string \n@attribute Tweet string\n@attribute Affect_Dimension string\n@attribute Intensity_Class '+class_value+' \n\n@data\n'
                    out.write(header_line)
                    firt_data_line=False

            parts=line.split("\t")
            if len(parts)==4:
         
                id=parts[0]
                tweet=parts[1].replace('\n','')
                tweet=re.sub('\'', '\"', tweet)
                emotion=parts[2]
                score=parts[3].strip() 
                score = score.split(":")[0] if score != "NONE" else "?"
                
                out_line='\''+id+'\',\''+tweet+'\','+'\''+emotion+'\','+score+'\n'
                out.write(out_line)
            else:
                print "Wrong format"
    

    f.close()  
    out.close()  
    
  
    
    
def create_multi_label_arff(input_file,output_file):
    """
    Creates an arff dataset for the e-c multi-label classification task
    """
  
    out=open(output_file,"w")  
    
    f=open(input_file, "rb")
    lines=f.readlines()
    
    
    
    
    in_header=True
    for line in lines:
        if(in_header):           
            header_line='@relation \''+os.path.basename(input_file)+': -C 11\'\n\n@attribute anger {0,1}\n@attribute anticipation {0,1}\n@attribute disgust {0,1}\n@attribute fear {0,1}\n@attribute joy {0,1}\n@attribute love {0,1}\n@attribute optimism {0,1}\n@attribute pessimism {0,1}\n@attribute sadness {0,1}\n@attribute surprise {0,1}\n@attribute trust {0,1} \n@attribute ID string \n@attribute Tweet string\n\n@data\n'
            out.write(header_line)
            in_header=False
        else:
            parts=line.split("\t")
            if len(parts)==13:
         
                id=parts[0]
                tweet=parts[1].replace('\n','')
                tweet=re.sub('\'', '\"', tweet)
                anger = parts[2] if parts[2] != "NONE" else "?"
                anticipation = parts[3] if parts[3] != "NONE" else "?"
                disgust = parts[4] if parts[4] != "NONE" else "?"
                fear = parts[5] if parts[5] != "NONE" else "?"
                joy = parts[6] if parts[6] != "NONE" else "?"
                love = parts[7] if parts[7] != "NONE" else "?"
                optimism = parts[8] if parts[8] != "NONE" else "?"
                pessimism = parts[9] if parts[9] != "NONE" else "?"
                sadness = parts[10] if parts[10] != "NONE" else "?"
                surprise = parts[11] if parts[11] != "NONE" else "?"
                trust = parts[12].strip() if parts[12].strip() != "NONE" else "?"
                          

                
                out_line=anger+','+anticipation+','+disgust+','+fear+','+joy+','+love+','+optimism+','+pessimism+','+sadness+','+surprise+','+trust+',\''+id+'\',\''+tweet+'\'\n'
                out.write(out_line)
            else:
                print "Wrong format"
    

    f.close()  
    out.close()      
    
    
    
def main(argv):
    """main method """   
    
    if len(argv)!=3:
        raise ValueError('Invalid number of parameters.')


    task_type=int(argv[0])
    input_file=argv[1]
    output_file=argv[2]

    if(task_type==1):
        create_reg_arff(input_file,output_file)
        
    elif(task_type==2):
        create_oc_arff(input_file,output_file)
        
    else:
        create_multi_label_arff(input_file,output_file)
   
        
if __name__ == "__main__":
    main(sys.argv[1:])    
    

