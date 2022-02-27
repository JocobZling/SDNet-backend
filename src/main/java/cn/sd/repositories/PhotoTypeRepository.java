package cn.sd.repositories;

import cn.sd.entities.PhotoType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface PhotoTypeRepository extends JpaRepository<PhotoType, Long> {
    @Query(value = "select * from photoType where typeList like %?1%", nativeQuery = true)
    List<PhotoType> findPhotoTypesByTypeListLike(String type);
}
