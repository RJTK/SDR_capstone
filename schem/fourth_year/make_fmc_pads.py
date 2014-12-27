'''
Generate all the locations for the pads for the fmc_lpc connector
'''

def main():
  '''
  '''
  FILE_NAME = 'fmc_lpc.kicad_mod'
  COLS = 40
  ROWS = 4

  with open(FILE_NAME, 'w') as f:
    f.write('(module FMC_LPC (layer F.Cu)\n'
            '  (at 0 0)\n'
            '  (fp_text reference FMC_LPC (at 0 -2.54) (layer F.SilkS)\n'
            '    (effects (font (size 1.524 1.524) (thickness 0.3048)))\n'
            '  )\n'
            '  (fp_text value VAL** (at 0 2.54) (layer F.SilkS)\n'
            '    (effects (font (size 1.524 1.524) (thickness 0.3048)))\n'
            '  )\n'
            '  (fp_line (start -27.89 7.34) (end -27.89 -7.34) (layer F.SilkS) '
            '(width 0.381))\n'

            '  (fp_line (start -27.89 7.34) (end 27.89 7.34) (layer F.SilkS) '
            '(width 0.381))\n'

            '  (fp_line (start 27.89 7.34) (end 27.89 -7.34) (layer F.SilkS) '
            '(width 0.381))\n'
            
            '  (fp_line (start -27.89 -7.34) (end 27.89 -7.34) (layer F.SilkS) '
            '(width 0.381))\n'
            )
    
    #The center of the coordinate system is the center of the fmc
    #Pin C40 is at the top left and pin H1 is at the bottom right
    #(Referring to the ASP-127797-01 datasheet)
    x_start = -24.765 #The x-coordinate for the left column
    x_coord = x_start
    x_inc = 1.270 #The spacing between the pads in the x direction
    y_start = -3.180 #The y-coordinate for the top row
    y_coord = -3.180 
    y_inc = 1.270 #The spacing between the pads in the y direction

    #There is extra space in the y direction between the 2nd and 3rd rows
    y_extra_inc = 2.55
    radius = 0.64

    row_letter = 'C'
    row_letter_extra = 2
    col_start = 40
    col_number = col_start
    for i in range(ROWS):
      for j in range(COLS):
        if(j == 0 and i == 2):
          y_coord += y_extra_inc
          row_letter = chr(ord(row_letter) + row_letter_extra)
        
        pad_name = row_letter + str(col_number)
        f.write('  (pad %s smd circle (at %f %f) '
                '(size %f %f)\n'
                
                '    (layers F.Cu B.Adhes F.Paste F.Mask)\n'
                '    (solder_mask_margin 0.2)\n'
                '    (solder_paste_margin 0.2)\n'
                '    (clearance 0.2)\n'
                '  )\n'
                %(pad_name, x_coord, y_coord, radius, radius)
                )
        x_coord += x_inc
        col_number -= 1
      
      x_coord = x_start
      y_coord += y_inc
      row_letter = chr(ord(row_letter) + 1)
      col_number = col_start
    f.write(')')
  return

if __name__ == '__main__':
  main()
