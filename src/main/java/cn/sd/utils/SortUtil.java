package cn.sd.utils;

import cn.sd.entities.Photo;
import cn.sd.entities.vo.PhotoDateAndSorce;
import cn.sd.entities.vo.PhotoTimeDisplay;
import lombok.*;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

//给Photo排序
@Setter
@Getter
@NoArgsConstructor
@Builder
public class SortUtil {
    public static List<Photo> SortPhoto(List<Photo> resultPhoto, boolean flag) {
        if (flag) {
            return SortPersonPhoto(resultPhoto);
        } else {
            return SortAllPhoto(resultPhoto);
        }
    }

    private static List<Photo> SortPersonPhoto(List<Photo> resultPhoto) {
        Collections.sort(resultPhoto, new Comparator<Photo>() {
            @Override
            public int compare(Photo o1, Photo o2) {
                String PersonScore1 = o1.getFaceScore();
                String PersonScore2 = o2.getFaceScore();

                Double Score1 = Double.parseDouble(PersonScore1);
                Double Score2 = Double.parseDouble(PersonScore2);

                Double diff = Score2 - Score1;

                if (diff > 0) {
                    return 1;
                } else if (diff < 0) {
                    return -1;
                }
                return 0;
            }
        });

        return resultPhoto;

    }

    private static List<Photo> SortAllPhoto(List<Photo> resultPhoto) {
        Collections.sort(resultPhoto, new Comparator<Photo>() {
            @Override
            public int compare(Photo o1, Photo o2) {
                String PersonScore1 = o1.getFaceScore();
                String PersonScore2 = o2.getFaceScore();

                String ColorScore1 = o1.getColorScore();
                String ColorScore2 = o2.getColorScore();


                Double Score1 = Double.parseDouble(PersonScore1) / 10 * 0.7 + Double.parseDouble(ColorScore1) * 0.3;
                Double Score2 = Double.parseDouble(PersonScore2) / 10 * 0.7 + Double.parseDouble(ColorScore2) * 0.3;

                Double diff = Score2 - Score1;

                if (diff > 0) {
                    return 1;
                } else if (diff < 0) {
                    return -1;
                }
                return 0;
            }
        });

        return resultPhoto;

    }

    public static String DateToStringDay(Date time) {
        DateFormat df = new SimpleDateFormat("yyyy-MM-dd");
        String strTime = df.format(time);
        return strTime;
    }

    public static List<Photo> SortTime(List<Photo> resultPhoto) {
        Collections.sort(resultPhoto, new Comparator<Photo>() {
            @Override
            public int compare(Photo o1, Photo o2) {
                Date time1 = o1.getCreateTime();
                Date time2 = o2.getCreateTime();

                String strTime1 = DateToStringDay(time1);
                String strTime2 = DateToStringDay(time2);

                int res = strTime2.compareTo(strTime1);
                return res;

            }
        });
        //System.out.println(resultPhoto);
        return resultPhoto;
    }

    public static List<Photo> GetNumPhotoBaseTime(List<Photo> resultPhoto, int num) {
        int len = resultPhoto.size();
        if (len <= num) {
            return resultPhoto;
        }
        List<Photo> SortTimePhoto = SortTime(resultPhoto);
        String CurTime = DateToStringDay(SortTimePhoto.get(0).getCreateTime());

        int index = 1;
        List<Photo> result = new ArrayList<Photo>();
        for (int i = 0; i < len; i++) {
            Photo tmp = SortTimePhoto.get(i);
            String TmpTime = DateToStringDay(tmp.getCreateTime());
            if (CurTime.compareTo(TmpTime) != 0) {
                CurTime = TmpTime;
                index++;
            }
            if (index > num) {
                break;
            }
            result.add(tmp);
        }
        return result;

    }

    public static List<Photo> GetTimePhoto(List<Photo> AllPhoto, String time) {
        int len = AllPhoto.size();
        List<Photo> result = new ArrayList<Photo>();
        for (int i = 0; i < len; i++) {
            Photo tmp = AllPhoto.get(i);
            String TmpTime = DateToStringDay(tmp.getCreateTime());

            if (TmpTime.compareTo(time) == 0) {
                result.add(tmp);
            }
        }
        return result;
    }

    public static List<List<Photo>> GetEveryTimePhoto(List<Photo> SortTimePhoto) {
        List<List<Photo>> result = new ArrayList<List<Photo>>();
        String curTime = DateToStringDay(SortTimePhoto.get(0).getCreateTime());

        int len = SortTimePhoto.size();
        int index = 0;

        while (index < len) {
            List<Photo> tmp = new ArrayList<Photo>();
            while (index < len && curTime.compareTo(DateToStringDay(SortTimePhoto.get(index).getCreateTime())) == 0) {
                tmp.add(SortTimePhoto.get(index));
                index++;

            }
            result.add(tmp);
            if (index < len) {
                curTime = DateToStringDay(SortTimePhoto.get(index).getCreateTime());
            }
        }
        return result;

    }

    public static List<PhotoDateAndSorce> SortPhotoDateAndSorce(List<PhotoDateAndSorce> resultPhoto) {
        Collections.sort(resultPhoto, new Comparator<PhotoDateAndSorce>() {
            @Override
            public int compare(PhotoDateAndSorce o1, PhotoDateAndSorce o2) {
                String Time1 = o1.getDate();
                String Time2 = o2.getDate();

                int res = Time2.compareTo(Time1);
                return res;
            }
        });

        return resultPhoto;

    }

    public static List<PhotoTimeDisplay> SortPhotoTimeDisplay(List<PhotoTimeDisplay> resultPhoto) {
        Collections.sort(resultPhoto, new Comparator<PhotoTimeDisplay>() {
            @Override
            public int compare(PhotoTimeDisplay o1, PhotoTimeDisplay o2) {
                String Time1 = o1.getDate();
                String Time2 = o2.getDate();

                int res = Time2.compareTo(Time1);
                return res;
            }
        });

        return resultPhoto;

    }


}
