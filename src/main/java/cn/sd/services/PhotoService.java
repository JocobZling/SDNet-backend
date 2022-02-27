package cn.sd.services;

import cn.sd.entities.Clustering;
import cn.sd.entities.FaceClustering;
import cn.sd.entities.Photo;
import cn.sd.entities.vo.PhotoDateAndSorce;
import cn.sd.entities.vo.PhotoDisplayVo;
import cn.sd.entities.vo.PhotoTimeDisplay;
import cn.sd.repositories.ClusteringRepository;
import cn.sd.repositories.FaceClusteringRepository;
import cn.sd.repositories.PhotoRepository;
import cn.sd.repositories.PhotoTypeRepository;
import cn.sd.utils.FileUtil;
import cn.sd.utils.OpenCVUtil;
import cn.sd.utils.SortUtil;
import com.alibaba.fastjson.JSONObject;
import com.chinamobile.cmss.sdk.ECloudDefaultClient;
import com.chinamobile.cmss.sdk.ECloudServerException;
import com.chinamobile.cmss.sdk.IECloudClient;
import com.chinamobile.cmss.sdk.http.constant.Region;
import com.chinamobile.cmss.sdk.http.signature.Credential;
import com.chinamobile.cmss.sdk.request.EngineImageClassifyDetectPostRequest;
import com.chinamobile.cmss.sdk.response.EngineImageClassifyDetectResponse;
import com.chinamobile.cmss.sdk.response.bean.EngineClassify;
import com.chinamobile.cmss.sdk.util.JacksonUtil;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.*;

@Service
public class PhotoService {

    @Value("${ak}")
    private String ak;

    @Value("${sk}")
    private String sk;

    @Value("${photoAddr}")
    private String photoAddr;

    @Value("${faceAddr}")
    private String faceAddr;

    @Value("${imageUrl}")
    private String imageUrl;

    private final PhotoRepository photoRepository;

    private final FaceClusteringRepository faceClusteringRepository;

    private final ClusteringRepository clusteringRepository;

    private final PhotoTypeRepository photoTypeRepository;

    private final FaceService faceService;

    private final UserCenterService userCenterService;


    public PhotoService(PhotoRepository photoRepository, FaceService faceService, FaceClusteringRepository faceClusteringRepository, ClusteringRepository clusteringRepository, PhotoTypeRepository photoTypeRepository, UserCenterService userCenterService) {
        this.faceClusteringRepository = faceClusteringRepository;
        this.photoRepository = photoRepository;
        this.faceService = faceService;
        this.clusteringRepository = clusteringRepository;
        this.photoTypeRepository = photoTypeRepository;
        this.userCenterService = userCenterService;
    }

    public void savePhotoToFile(MultipartFile[] files, Long userId) throws Exception {
        String airSetId = userCenterService.findAirSetIdByUserId(userId);
        assert files != null;
        for (MultipartFile file : files) {
            String photoSavePosition = imageUrl + Objects.requireNonNull(FileUtil.saveFile(file, photoAddr)).get(0);
            String photoAddrPath = Objects.requireNonNull(FileUtil.saveFile(file, photoAddr)).get(1);
            Photo photo = new Photo();
            String colorScore = OpenCVUtil.calculateColorScore(photoAddrPath);
            String image = FileUtil.encryptToBase64(photoAddrPath);
            JSONObject response = faceService.getFaceDetection(photoAddrPath);
            double faceNum;
            if (response == null)
                faceNum = 0;
            else
                faceNum = Double.parseDouble(String.valueOf(response.get("faceNum")));
            photo.setFaceScore(String.valueOf(faceNum * 10));
            photo.setColorScore(colorScore);
            photo.setPosition(photoSavePosition);
            photo.setUserId(userId);
            photoRepository.save(photo);
            Long photoId = photo.getId();
            if (faceNum > 0 && faceNum <= 10) {
                photo.setType("人物");
                ArrayList<?> faceDetectDetailList = (ArrayList<?>) response.get("faceDetectDetailList");
                List<FaceClustering> faceClusteringList = faceService.getAllClusteringOneFaceByUserId(userId);
                //对该位置人脸进行切割
                // OpenCVUtil.findAndCutFace(photoAddrPath, faceAddrPath);
                faceDetectDetailList.stream().map(faceDetect -> (Map<?, ?>) faceDetect).map(detect -> (Map<?, ?>) detect.get("faceDectectRectangleArea")).forEach(faceDetectRectangleArea -> {
                    double lx = Double.parseDouble(String.valueOf(faceDetectRectangleArea.get("upperLeftX")));
                    double ly = Double.parseDouble(String.valueOf(faceDetectRectangleArea.get("upperLeftY")));
                    double rx = Double.parseDouble(String.valueOf(faceDetectRectangleArea.get("lowerRightX")));
                    double ry = Double.parseDouble(String.valueOf(faceDetectRectangleArea.get("lowerRightY")));
                    String faceName = userId + "_" + UUID.randomUUID().toString().replaceAll("-", "").substring(0, 10) + "_face.png";
                    String faceAddrPath = faceAddr + faceName;
                    try {
                        OpenCVUtil.cutPhotoFace(lx, ly, rx, ry, photoAddrPath, faceAddrPath);
                        //positions.add(faceAddrPath);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    FaceClustering faceClustering = new FaceClustering();
                    faceClustering.setPosition(imageUrl + faceName);
                    faceClustering.setPhotoId(photo.getId());
                    faceClustering.setUserId(userId);
                    if (faceClusteringList.size() > 0) {
                        Map<?, ?> result = faceService.searchFace(faceAddrPath, Integer.parseInt(airSetId));
                        if (result == null) {
                            return;
                        }
                        if (Double.parseDouble(String.valueOf(result.get("confidence"))) > 0.6) {
                            int faceId = Double.valueOf(String.valueOf(result.get("faceId"))).intValue();
                            Long clusteringId = faceService.findFaceClusteringByFaceId(faceId).getClusteringId();
                            faceClustering.setClusteringId(clusteringId);
                            String newFaceId = faceService.addFaceToAirSet(Integer.parseInt(airSetId), faceAddrPath, faceName);
                            Integer airFaceId = Double.valueOf(newFaceId).intValue();
                            faceClustering.setAirFaceId(String.valueOf(airFaceId));
                        } else {
                            // 置信度不够 -> 设置为新的
                            String faceId = faceService.addFaceToAirSet(Integer.parseInt(airSetId), faceAddrPath, faceName);
                            if (faceId.equals("error"))
                                return;
                            Integer airFaceId = Double.valueOf(faceId).intValue();
                            faceClustering.setAirFaceId(String.valueOf(airFaceId));
                            Clustering clustering = new Clustering();
                            clustering.setUserId(userId);
                            clusteringRepository.save(clustering);
                            faceClustering.setClusteringId(clustering.getId());
                        }
                    } else {
                        String faceId = faceService.addFaceToAirSet(Integer.parseInt(airSetId), faceAddrPath, faceName);
                        if (faceId.equals("error"))
                            return;
                        Integer airFaceId = Double.valueOf(faceId).intValue();
                        faceClustering.setAirFaceId(String.valueOf(airFaceId));
                        Clustering clustering = new Clustering();
                        clustering.setUserId(userId);
                        clusteringRepository.save(clustering);
                        faceClustering.setClusteringId(clustering.getId());
                    }
                    faceClusteringRepository.save(faceClustering);
                });
            } else if (faceNum > 10) {
                photo.setFaceScore(String.valueOf(faceNum * 10 * 0.4));
                photo.setType("群像");
            } else {
                String type = getPhotoType(getPhotoType(userId, image));
                photo.setType(type);
            }
            photoRepository.save(photo);
        }
    }

    public String getPhotoType(long userId, String image) throws IOException {
        //企业云账户：请申请
        Credential credential = new Credential(ak, sk);

        //初始化ECloud client,Region 为部署资源池OP网关地址枚举类，可自行扩展
        IECloudClient ecloudClient = new ECloudDefaultClient(credential, Region.POOL_SZ);

        //待定义产品request
        try {
            //通用物品识别
            EngineImageClassifyDetectPostRequest request = new EngineImageClassifyDetectPostRequest();
            //图片base64 ，注意不要包含 {data:image/jpeg;base64,}
            request.setImage(image);
            request.setUserId(String.valueOf(userId));
            //通用物品检测
            EngineImageClassifyDetectResponse response = ecloudClient.call(request);
            if ("OK".equals(response.getState())) {
                //通用物品检测
                List<EngineClassify> body = response.getBody();
                return JacksonUtil.toJson(body).split(":")[1].split("\"")[1];
            }
        } catch (IOException | ECloudServerException | IllegalAccessException e) {
            //todo exception process...
            e.printStackTrace();
            return null;
        }
        return null;
    }

    public List<PhotoTimeDisplay> findAllDaysPhotosByUserId(Long userId) {
        List<Photo> photoList = photoRepository.findPhotoByUserIdOrderByCreateTimeDesc(userId);
        HashSet<String> date = new HashSet<>();
        List<PhotoTimeDisplay> result = new ArrayList<>();
        photoList.forEach(photo -> {
            date.add(photo.getCreateTime().toString().split(" ")[0]);
        });
        for(String d : date)
        {
            List<Photo> tmpPhotoList = new ArrayList<>();
            PhotoTimeDisplay photoTimeDisplay = new PhotoTimeDisplay();
            for(Photo photo : photoList)
            {
                if(d.compareTo(photo.getCreateTime().toString().split(" ")[0]) == 0)
                {
                    tmpPhotoList.add(photo);
                }
            }
            photoTimeDisplay.setDate(d);
            photoTimeDisplay.setPhotoDisplayVoList(FaceService.getPhotoDisplayVos(tmpPhotoList, photoAddr));
            result.add(photoTimeDisplay);
        }

        SortUtil.SortPhotoTimeDisplay(result);

        return result;



//        photoList.forEach(photo -> {
//            date.add(photo.getCreateTime().toString().split(" ")[0]);
//        });
//        date.forEach(s -> {
//            PhotoTimeDisplay phototimedisplay = new PhotoTimeDisplay();
//            phototimedisplay.setDate(s);
////            List<Photo> photos = new ArrayList<>();
////            photoList.forEach(photo -> {
////                if (photo.getCreateTime().toString().split(" ")[0].equals(s)) photos.add(photo);
////            });
//            phototimedisplay.setPhotoDisplayVo();
//            photoDateAndSorce.setPhotoList(photos);
//            result.add(photoDateAndSorce);
//        });
        //这里返回的是List<PhotoDisplayVo>
       // FaceService.getPhotoDisplayVos(photoList, photoAddr);
    }

    public List<PhotoDisplayVo> findOneDaysAllPhotosByUserId(Long userId, String date) {
        List<Photo> photoList = photoRepository.findPhotoByUserIdAndCreateTime(userId, date);
        return faceService.getPhotoDisplayList(photoList);
    }

    public List<PhotoDisplayVo> findOneTypeAllPhotosByUserId(String type, Long userId) {
        List<Photo> photoList = photoRepository.findPhotoByTypeAndUserId(type, userId);
        return faceService.getPhotoDisplayList(photoList);
    }


    public List<PhotoDateAndSorce> findAllTimesEightPhotoByUserId(Long userId) {
        List<Photo> photoList = photoRepository.findPhotoByUserIdOrderByCreateTimeDesc(userId);
        HashSet<String> date = new HashSet<>();
        List<PhotoDateAndSorce> result = new ArrayList<>();
        photoList.forEach(photo -> {
            date.add(photo.getCreateTime().toString().split(" ")[0]);
        });
        date.forEach(s -> {
            PhotoDateAndSorce photoDateAndSorce = new PhotoDateAndSorce();
            photoDateAndSorce.setDate(s);
            List<Photo> photos = new ArrayList<>();
            photoList.forEach(photo -> {
                if (photo.getCreateTime().toString().split(" ")[0].equals(s)) photos.add(photo);
            });
            photoDateAndSorce.setPhotoList(photos);
            result.add(photoDateAndSorce);
        });
        //对result按照时间排序
        SortUtil.SortPhotoDateAndSorce(result);

        return result;
    }

    public List<HashMap<String, String>> findAllTypeOnePhotoByUserId(Long userId) {
        List<HashMap<String, String>> photoList = new ArrayList<>();
        HashMap<String, String> qiTaTypePhoto = new HashMap<>();
        photoRepository.findAllTypeOnePhotoByUserId(userId).forEach(photo -> {
            HashMap<String, String> oneTypePhoto = new HashMap<>();
            if(photo.getType().compareTo("其他")==0)
            {

                qiTaTypePhoto.put("type", photo.getType());
                qiTaTypePhoto.put("position", photo.getPosition());

            }
            else
            {
                oneTypePhoto.put("type", photo.getType());
                oneTypePhoto.put("position", photo.getPosition());
                photoList.add(oneTypePhoto);
            }
        });
        photoList.add(qiTaTypePhoto);
        return photoList;
    }

    private String getPhotoType(String input) {
        String[] typeList = input.split(",");
        String mainType = "其他";
        for (String type : typeList) {
            if (!mainType.equals("其他")) return mainType;
            if (photoTypeRepository.findPhotoTypesByTypeListLike(type).size() > 0) {
                mainType = photoTypeRepository.findPhotoTypesByTypeListLike(type).get(0).getMainType();
            }
        }
        return mainType;
    }
}
