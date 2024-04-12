# Jerry Xia 260917329

I'd like to waive the three-day penalty for this assignment.

Additionally, I've added all files in the provided code in my submission since I created `.json` and `.obj` files in case you need to run it.

I've also included my submission for the best in show!

Some additional points:

- For `TorusMesh.json`, I changed the path from `"./meshes/torus.obj"` to `"../meshes/torus.obj"`
- I removed `"light3"` so the image would output - it wasn't before for some reason, as stated by this [Ed post](https://edstem.org/us/courses/51599/discussion/4698960).

### Images

Images are sorted into two directories `img/base` and `img/extra`.

The extra features are:

- Mirroring
- Refraction
- Motion blur
- Depth of field blur
- Area lights, i.e. soft shadows -- I included several images to show these.
- Quadrics

Since meshes take a considerable time to render, I split my novel scene into two parts: `NovelScene1.png` and `NovelScene2.png`.

- `NovelScene1.png` shows off mesh rendering with a bunny in the back, as well as soft shadows.
- `NovelScene2.png` shows off other features, such as cube rendering, a more noticeable depth of field blur, as well as motion blur.
- These features are further annotated in `NovelScene1Annotated.png` and `NovelScene2Annotated.png`.
