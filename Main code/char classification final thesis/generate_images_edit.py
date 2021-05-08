import os
import numpy as np
import uuid
import random



PATH_TO_LIGHT_BACKGROUND ='light_backgrounds/'
PATH_TO_DARK_BACKGROUND ='dark_backgrounds/'
PATH_TO_FONT_FILE ='fonts/'
OUTPUT_DIRC ='output/'
NUMber_IMAGES_PER_CLASS =10
#FIles from directory
def get_files_from_dir(dircname):
  list_file =(os.listdir(dircname))
  list_file =[dircname + x for x in list_file]
  return list_file
#random perspective distortion
def get_distort_args():
  amounts =5
  hundreds_minus_amount= 100 - amounts
  return '\'0,0 ' +str(np.random.randint(0 ,amounts))+ ',' +str(np.random.randint(0 ,amounts))+ ' 100,0 '  +str(np.random.randint(hundreds_minus_amount,100))+ ',' +str(np.random.randint(0, amounts))+ ' 0,100 '  +str(np.random.randint(0,amounts))+ ',' +str(np.random.randint(hundreds_minus_amount,100))+ ' 100,100 '  +str(np.random.randint(hundreds_minus_amount,100))+ ',' +str(np.random.randint(hundreds_minus_amount,100))+ '\''

#randomly extracts 32x32 region of an image
def create_random_crops(image_filenames,num_crop,out_dirc):
  dimn= os.popen('convert ' +image_filenames+ ' -ping -format "%w %h" info:').read()
  im_widths= int(dimn[0])
  im_heights= int(dimn[1])
  dimn= dimn.split()
  
  for i in range(0,num_crop):
    #randomly select cropping image
    x= random.randint(0,im_widths - 32)
    y= random.randint(0,im_heights - 32)
    outfiles= uuid.uuid4().hex + '.jpg'
    commands= "magick convert "+image_filenames +" -crop 32x32"+"+"+str(x)+"+"+str(y)+" " +os.path.join(out_dirc,outfiles)
    os.system(str(commands))

def generate_crops(file_listt, dircname):
  if not os.path.isdir(dircname):
    os.mkdir(dircname)
    for f in file_listt:
      create_random_crops(f,10,dircname)

char_listt= []
for i in range(65,65+26):
  char_listt.append(chr(i))
#digits
for j in range(48,48+10):
  char_listt.append(chr(j))
#light font colorrs
colorr_light= ['white','gray','silver','lime','yellow','aqua']
#light dark colorrs
colorr_dark= ['black','maroon','green','purple','blue','red']

#light backgrounds
light_backgrounds= get_files_from_dir(PATH_TO_LIGHT_BACKGROUND)

#dark backgrounds
dark_backgrounds= get_files_from_dir(PATH_TO_DARK_BACKGROUND)
#font files
list_file_fontt= get_files_from_dir(PATH_TO_FONT_FILE)
light_backgrounds_crops_dirc= 'light_backgrounds_crops/'
dark_backgrounds_crops_dirc= 'dark_backgrounds_crops/'
generate_crops(light_backgrounds,light_backgrounds_crops_dirc)
generate_crops(dark_backgrounds,dark_backgrounds_crops_dirc)
#files in the directory
light_backgrounds= get_files_from_dir(light_backgrounds_crops_dirc)
dark_backgrounds= get_files_from_dir(dark_backgrounds_crops_dirc)
#all backgrounds
all_backgrounds= [dark_backgrounds, light_backgrounds]

for i in range(0,len(char_listt)):
  char= char_listt[i]
  char_OUTPUT_DIRCs= OUTPUT_DIRC + str(char) + "/"
	
  if not os.path.exists(char_OUTPUT_DIRCs):
    os.makedirs(char_OUTPUT_DIRCs)

  print("Generating data!! " +char_OUTPUT_DIRCs)

  for j in range(0,NUMber_IMAGES_PER_CLASS):

    path= random.choice(all_backgrounds)
  
    list_filernds= random.choice(path)
    list_rfos= random.choice(list_file_fontt)
    
    distort_args= get_distort_args()
 
    blur= random.randint(0,3)
    
    noise= random.randint(0,5)
   
    x= str(random.randint(-3,3))
    y= str(random.randint(-3,3))
    
    if path== all_backgrounds[0] :
      colorr= random.choice(colorr_light)
    else:
      colorr= random.choice(colorr_dark)

    commands=  "magick convert " +str(list_filernds)+ " -fill "+str(colorr)+" -font "+ \
            str(list_rfos)+ " -weight 200 -pointsize 24 -distort Perspective "+str(distort_args)+" "+"-gravity center"+" -blur 0x" +str(blur) \
+" -evaluate Gaussian-noise "+str(noise)+" " +" -annotate +" +x+ "+"+y+" "+str(char_listt[i])+ " " +char_OUTPUT_DIRCs+ "output_file"+str(i)+str(j)+".jpg"
		
    #uncommnt
    #print(commands)
    os.system(str(commands))