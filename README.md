# stack_img

Does not support video file input. Images to be stacked must be placed
into a folder, the folder is passed to the script.

## Single image stack
`python -m stack_img stack <FOLDER>`

## Animate the stacking
`python -m stack_img animate <FOLDER>`

## Stack every Nth image
`python -m stack_img animate <FOLDER> --nframe 2`

## Limit the number of a frames in a stack (animated tail)
The larger the number, the longer the movie will take to compile

`python -m stack_img animate <FOLDER> --frange 60`

## Combininations are supported
Mix and match the range of frames and Nth frame arguments

`python -m stack_img animate <FOLDER> --frange 5 --nframe 2`