function IM = fetchExperimentData(name, env, isRaw)
    if isRaw
        rootDir = strcat('D:\jyu\converted_raw_data\', env, '\');
    else
        rootDir = strcat('D:\jyu\converted_processed_data\', env, '\');
    end
    
    switch name
        case 'plastic ice cream measurement'
            IM(1).imdir  = rootDir;
            IM(1).imname = '200-int60-dim3-cal60';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '201-int3-dim3-cal60';
            IM(2).key    = 'data';

            IM(3).imdir  = rootDir;
            IM(3).imname = '202-int1-dim3-cal60';
            IM(3).key    = 'data';
        case 'book with colored stripes'
            IM(1).imdir  = rootDir;
            IM(1).imname = '216-int60-dim3-cal60';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '217-int3-dim3-cal60';
            IM(2).key    = 'data';

            IM(3).imdir  = rootDir;
            IM(3).imname = '218-int10-dim3-cal60';
            IM(3).key    = 'data';

            IM(4).imdir  = rootDir;
            IM(4).imname = '219-int1-dim3-cal60';
            IM(4).key    = 'data';

            IM(5).imdir  = rootDir;
            IM(5).imname = '220-int5-dim3-cal60';
            IM(5).key    = 'data';
        case 'roald dahl billy and the minpins book'
            IM(1).imdir  = rootDir;
            IM(1).imname = '221-int60-dim3-cal60';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '222-int10-dim3-cal60';
            IM(2).key    = 'data';

            IM(3).imdir  = rootDir;
            IM(3).imname = '223-int5-dim3-cal60';
            IM(3).key    = 'data';
        case 'this is finland book'
            IM(1).imdir  = rootDir;
            IM(1).imname = '229-int60-dim3-cal60';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '230-int10-dim3-cal60';
            IM(2).key    = 'data';

            IM(3).imdir  = rootDir;
            IM(3).imname = '231-int5-dim3-cal60';
            IM(3).key    = 'data';

            IM(4).imdir  = rootDir;
            IM(4).imname = '232-int3-dim3-cal60';
            IM(4).key    = 'data';
        case 'Moomin cards set 1'
            IM(1).imdir  = rootDir;
            IM(1).imname = '233-int60-dim3-cal60';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '234-int10-dim3-cal60';
            IM(2).key    = 'data';

            IM(3).imdir  = rootDir;
            IM(3).imname = '235-int5-dim3-cal60';
            IM(3).key    = 'data';

            IM(4).imdir  = rootDir;
            IM(4).imname = '236-int3-dim3-cal60';
            IM(4).key    = 'data';

            IM(5).imdir  = rootDir;
            IM(5).imname = '237-int3-dim3-cal60';
            IM(5).key    = 'data';

            IM(6).imdir  = rootDir;
            IM(6).imname = '238-int1-dim3-cal60';
            IM(6).key    = 'data';

            IM(7).imdir  = rootDir;
            IM(7).imname = '239-int2-dim3-cal60';
            IM(7).key    = 'data';
        case 'wooden chess pieces with calibrator'
            IM(1).imdir  = rootDir;
            IM(1).imname = '285-int30-dim5-cal30';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '286-int10-dim5-cal10';
            IM(2).key    = 'data';
        case 'plastic toys with calibrator on checkerboard'
            IM(1).imdir  = rootDir;
            IM(1).imname = '296-int60-dim3-cal60';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '297-int10-dim3-cal10';
            IM(2).key    = 'data';
        case 'scissors with calibrator on checkerboard'
            IM(1).imdir  = rootDir;
            IM(1).imname = '299-int60-dim3-cal60';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '300-int10-dim3-cal10';
            IM(2).key    = 'data';

        case 'Parking lot- scene 1'
            IM(1).imdir  = rootDir;
            IM(1).imname = '333-intunk-dimnoDim-calunk';
            IM(1).key    = 'data';
			
			IM(2).imdir  = rootDir;
            IM(2).imname = '334-int2-dimnoDim-cal2';
            IM(2).key    = 'data';
			
			IM(3).imdir  = rootDir;
            IM(3).imname = '335-intunk-dimnoDim-calunk';
            IM(3).key    = 'data';
			
			IM(4).imdir  = rootDir;
            IM(4).imname = '336-int10-dimnoDim-cal10';
            IM(4).key    = 'data';
			
			IM(5).imdir  = rootDir;
            IM(5).imname = '337-int30-dimnoDim-cal30';
            IM(5).key    = 'data';
			
			IM(6).imdir  = rootDir;
            IM(6).imname = '338-int2-dimnoDim-cal2';
            IM(6).key    = 'data';
			
			IM(7).imdir  = rootDir;
            IM(7).imname = '339-int5-dimnoDim-cal5';
            IM(7).key    = 'data';
			
			IM(8).imdir  = rootDir;
            IM(8).imname = '340-int10-dimnoDim-cal10';
            IM(8).key    = 'data';

        case 'Parking lot- scene 2'
            IM(1).imdir  = rootDir;
            IM(1).imname = '343-int5-dimnoDim-cal5';
            IM(1).key    = 'data';
			
			IM(2).imdir  = rootDir;
            IM(2).imname = '344-int10-dimnoDim-cal10';
            IM(2).key    = 'data';
			
			IM(3).imdir  = rootDir;
            IM(3).imname = '345-int20-dimnoDim-cal20';
            IM(3).key    = 'data';
			
			IM(4).imdir  = rootDir;
            IM(4).imname = '346-int20-dimnoDim-cal20';
            IM(4).key    = 'data';
			
			IM(5).imdir  = rootDir;
            IM(5).imname = '347-int20-dimnoDim-cal20';
            IM(5).key    = 'data';
			
			IM(6).imdir  = rootDir;
            IM(6).imname = '348-int10-dimnoDim-cal10';
            IM(6).key    = 'data';
			
			IM(7).imdir  = rootDir;
            IM(7).imname = '349-int5-dimnoDim-cal5';
            IM(7).key    = 'data';

			IM(8).imdir  = rootDir;
            IM(8).imname = '350-int20-dimnoDim-cal20';
            IM(8).key    = 'data';
            
            IM(9).imdir  = rootDir;
            IM(9).imname = '351-int5-dimnoDim-cal5';
            IM(9).key    = 'data';
        
        case 'Street'
            IM(1).imdir  = rootDir;
            IM(1).imname = '366-int14-dim3-cal14';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '367-int20-dim3-cal20';
            IM(2).key    = 'data';

            IM(3).imdir  = rootDir;
            IM(3).imname = '368-int20-dim3-cal20';
            IM(3).key    = 'data';

            IM(4).imdir  = rootDir;
            IM(4).imname = '369-int40-dim3-cal40';
            IM(4).key    = 'data';

            IM(5).imdir  = rootDir;
            IM(5).imname = '370-int14-dim3-cal14';
            IM(5).key    = 'data';

            IM(6).imdir  = rootDir;
            IM(6).imname = '371-int5-dim3-cal5';
            IM(6).key    = 'data';

            IM(7).imdir  = rootDir;
            IM(7).imname = '372-int10-dim3-cal10';
            IM(7).key    = 'data';

            IM(8).imdir  = rootDir;
            IM(8).imname = '373-int10-dim3-cal10';
            IM(8).key    = 'data';

            IM(9).imdir  = rootDir;
            IM(9).imname = '374-int15-dim3-cal15';
            IM(9).key    = 'data';

            IM(10).imdir  = rootDir;
            IM(10).imname = '375-int30-dim3-cal30';
            IM(10).key    = 'data';

            IM(11).imdir  = rootDir;
            IM(11).imname = '376-int8-dim3-cal8';
            IM(11).key    = 'data';

            IM(12).imdir  = rootDir;
            IM(12).imname = '377-int60-dim3-cal60';
            IM(12).key    = 'data';

            IM(13).imdir  = rootDir;
            IM(13).imname = '378-int300-dim3-cal300';
            IM(13).key    = 'data';

            IM(14).imdir  = rootDir;
            IM(14).imname = '379-int300-dim3-cal300';
            IM(14).key    = 'data';

            IM(15).imdir  = rootDir;
            IM(15).imname = '380-int35-dim3-cal35';
            IM(15).key    = 'data';

            IM(16).imdir  = rootDir;
            IM(16).imname = '381-int35-dim3-cal35';
            IM(16).key    = 'data';

            IM(17).imdir  = rootDir;
            IM(17).imname = '382-int180-dim3-cal180';
            IM(17).key    = 'data';

            IM(18).imdir  = rootDir;
            IM(18).imname = '383-int450-dim3-cal450';
            IM(18).key    = 'data';

            IM(19).imdir  = rootDir;
            IM(19).imname = '384-int20-dim3-cal20';
            IM(19).key    = 'data';

            IM(20).imdir  = rootDir;
            IM(20).imname = '385-int500-dim3-cal500';
            IM(20).key    = 'data';

        case 'Crosswalk'
            IM(1).imdir  = rootDir;
            IM(1).imname = '387-int500-dim3-cal500';
            IM(1).key    = 'data';

            IM(2).imdir  = rootDir;
            IM(2).imname = '388-int30-dim3-cal30';
            IM(2).key    = 'data';

            IM(3).imdir  = rootDir;
            IM(3).imname = '389-int10-dim3-cal10';
            IM(3).key    = 'data';

            IM(4).imdir  = rootDir;
            IM(4).imname = '390-int20-dim3-cal20';
            IM(4).key    = 'data';

            IM(5).imdir  = rootDir;
            IM(5).imname = '391-int30-dim3-cal30';
            IM(5).key    = 'data';

            IM(6).imdir  = rootDir;
            IM(6).imname = '392-int30-dim3-cal30';
            IM(6).key    = 'data';

            IM(7).imdir  = rootDir;
            IM(7).imname = '393-int60-dim3-cal60';
            IM(7).key    = 'data';

            IM(8).imdir  = rootDir;
            IM(8).imname = '394-int30-dim3-cal30';
            IM(8).key    = 'data';

            IM(9).imdir  = rootDir;
            IM(9).imname = '395-int60-dim3-cal60';
            IM(9).key    = 'data';

            IM(10).imdir  = rootDir;
            IM(10).imname = '396-int100-dim3-cal100';
            IM(10).key    = 'data';

        otherwise
            error('Unknown experiment name');
    end
end