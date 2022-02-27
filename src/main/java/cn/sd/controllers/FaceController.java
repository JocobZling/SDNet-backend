package cn.sd.controllers;

import cn.sd.services.FaceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping(value = "/api/face")
public class FaceController {

    private final FaceService faceService;


    @Autowired
    public FaceController(FaceService faceService) {
        this.faceService = faceService;
    }

    @GetMapping(value = "/all/one/{userId}")
    public ResponseEntity<?> getAllFaceOnePhoto(@PathVariable Long userId) {
        return ResponseEntity.ok(faceService.getAllClusteringOneFaceByUserId(userId));
    }

    @GetMapping(value = "/eight/one/{userId}")
    public ResponseEntity<?> getEightFaceOnePhoto(@PathVariable Long userId) {
        return ResponseEntity.ok(faceService.getEightClusteringOneFaceByUserId(userId));
    }
    // 返回人脸位置 name 以及list
    @GetMapping(value = "/oneKlass/{clusteringId}/all/{userId}")
    public ResponseEntity<?> getOneKlassAllPhoto(@PathVariable Long userId, @PathVariable Long clusteringId) {
        return ResponseEntity.ok(faceService.findOneKlassAllPhotoByUserIdAndClusteringId(userId, clusteringId));
    }

    // 更新clusterName
    @PutMapping(value = "/update/{clusteringId}/clusteringName/{userId}")
    public ResponseEntity<?> updateClusterName(@PathVariable Long userId, @PathVariable Long clusteringId,@RequestBody String name)
    {
        if(faceService.updateClusteringName(userId,clusteringId,name))
            return ResponseEntity.ok("更新成功");
        else return new ResponseEntity<>("更新失败", HttpStatus.BAD_REQUEST);
    }


}
