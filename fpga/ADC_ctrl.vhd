library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity ADC_ctrl is
  port ( D11 : in STD_LOGIC;
         D10 : in STD_LOGIC;
         D09 : in STD_LOGIC;
         D08 : in STD_LOGIC;
         D07 : in STD_LOGIC;
         D06 : in STD_LOGIC;
         D05 : in STD_LOGIC;
         D04 : in STD_LOGIC;
         D03 : in STD_LOGIC;
         D02 : in STD_LOGIC;
         D01 : in STD_LOGIC;
         D00 : in STD_LOGIC;
         CLK_IN : in STD_LOGIC;
         RST : in STD_LOGIC;
         DATA_OUT : out STD_LOGIC_VECTOR(11 downto 0);
         CLK_OUT : out STD_LOGIC);
end ADC_ctrl;

architecture Behavioral of ADC_ctrl is

begin

clocking : process(CLK_IN)
begin
  CLK_OUT <= CLK_IN;
end process clocking;

reg_data_recv : process(CLK_IN)
begin
  if(CLK_IN'Event and CLK_IN = '1') then
    if (RST = '0') then
      DATA_OUT <= (others => '0');
    else
      DATA_OUT <= (D11 & D10 & D09 & D08 & D07 & D06 & D05 & D04 & D03 & D02 & D01 & D00);
    end if;
  end if;
end process reg_data_recv;

end Behavioral;

