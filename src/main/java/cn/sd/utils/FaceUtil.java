package cn.sd.utils;

import com.chinamobile.bcop.api.sdk.ai.core.constant.Region;
import com.chinamobile.bcop.api.sdk.ai.facebody.AiFaceBody;

public class FaceUtil {

    private static final String ak = "CIDC-AK-37991b47-300a-4fd3-9252-66b826df4a6a";
    private static final String sk = "CIDC-SK-dc42c53a-d87f-4310-97f9-fca060855eee";

    private static AiFaceBody aiFaceBody;

    public static AiFaceBody getInstance() {
        if (aiFaceBody == null) {
            aiFaceBody = new AiFaceBody(Region.POOL_CS, ak, sk);
        }
        return aiFaceBody;
    }
}
