CKPTS = [

    dict(
        path="/home/chunyl/azure_mount/yuhengdb/fine_tune_ldm/version5_branch6_output/GoldG+SBU+CC3M+CC12M+O365/second_stage_drop_both/tag01/checkpoint_00450001.pth",
        feature_type=['before','after_reproject'],
        save_folder_name="v5b6_drop_both",
    ),


    # dict(
    #     path="/home/v-yuhengli/blobfuse/output/fine_tune_ldm/version5_branch6_output/GoldG+SBU+CC3M+CC12M+O365/second_stage_drop_none/tag00/checkpoint_00165001.pth",
    #     feature_type=['before','after_reproject'],
    #     save_folder_name="v5b6_drop_none",
    # ),





]



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #








    # if meta["has_image_mask"] == 0:
    #     image_embeddings = text_embeddings
    # if meta["has_text_mask"] == 0:
    #     text_embeddings = image_embeddings

    # out = {
    #     "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
    #     "masks" : masks.unsqueeze(0).repeat(batch,1),
    #     "text_masks" : masks.unsqueeze(0).repeat(batch,1),
    #     "image_masks" : masks.unsqueeze(0).repeat(batch,1),
    #     "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
    #     "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    # }







META = [


    dict(
        prompt = "a teddy bear sitting next to a red bird",
        phrases = ['a teddy bear', 'a red bird'],
        images = ['images/teddy.jpg', 'images/red_bird.jpg'],
        locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8]   ],
        alpha_type = [1.0, 0, 0.0], 
        has_text_mask = 1,  
        has_image_mask = 0,  
        save_folder_name="teddy_bird_1_1"
    ),


    # dict(
    #     prompt = "a teddy bear sitting next to a bird",
    #     phrases = ['a teddy bear', 'a bird'],
    #     images = ['images/teddy.jpg', 'images/red_bird.jpg'],
    #     locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8]   ],
    #     alpha_type = [1.0, 0, 0.0], 
    #     has_text_mask = 1,  
    #     has_image_mask = 1,  
    #     save_folder_name="teddy_bird_1_1"
    # ),


    # dict(
    #     prompt = "a teddy bear sitting next to a bird",
    #     phrases = ['a teddy bear', 'a bird'],
    #     images = ['images/teddy.jpg', 'images/red_bird.jpg'],
    #     locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8]   ],
    #     alpha_type = [0.5, 0, 0.5], 
    #     has_text_mask = 1,  
    #     has_image_mask = 0,  
    #     save_folder_name="teddy_bird_1_0"
    # ),

    # dict(
    #     prompt = "",
    #     phrases = ['a teddy bear', 'an umbrella'],
    #     images = ['images/teddy.jpg', 'images/umbrella.png'],
    #     locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8]   ],
    #     alpha_type = [1.0, 0, 0.0], 
    #     has_text_mask = 1,  
    #     has_image_mask = 1,  
    #     save_folder_name="empty_teddy_umbrella_1_1"
    # ),

    # dict(
    #     prompt = "hello kitty and bird hybrid",
    #     phrases = ['a hello kitty', 'a hello kitty'],
    #     images = ['images/red_bird.jpg', 'images/red_bird.jpg'],
    #     locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8]   ],
    #     has_text_mask = 1,  
    #     has_image_mask = 1,  
    #     save_folder_name="hello+bird_1_1"
    # ),

    # dict(
    #     prompt = "hello kitty and teddy bear hybrid",
    #     phrases = ['a hello kitty', 'a hello kitty'],
    #     images = ['images/teddy.jpg', 'images/teddy.jpg'],
    #     locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8]   ],
    #     has_text_mask = 1,  
    #     has_image_mask = 1,  
    #     save_folder_name="hello+teddy_1_1"
    # ),

    # dict(
    #     prompt = "bird and hello kitty hybrid",
    #     phrases = ['a bird', 'a bird'],
    #     images = ['images/hello.jpg', 'images/hello.jpg'],
    #     locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8]   ],
    #     alpha_type = [1.0, 0, 0.0], 
    #     has_text_mask = 1,  
    #     has_image_mask = 0.5,  
    #     save_folder_name="bird+hello_1_1"
    # ),



    # dict(
    #     prompt = "a deer standing in front of a brick house in the woods, anime, oil painting, high resolution, cottagecore, ghibli inspired, 4k",
    #     phrases = ['a deer'],
    #     images = ['images/sky.jpg'],
    #     locations = [ [0.0,0.5,0.5,0.9] ],
    #     alpha_type = [1, 0, 0],  
    #     has_text_mask = 1,  
    #     has_image_mask = 1,  
    #     save_folder_name="deer_sky"
    # ),


    # dict(
    #     prompt = "A woman sitting in a restaurant with a slice of pizza in front of her",
    #     phrases = ['dining table', 'pizza', 'person', 'wall', 'car', 'paper', 'chair', 'window', 'bottle', 'cup'],
    #     images = ['images/hello.jpg','images/hello.jpg','images/hello.jpg','images/hello.jpg','images/hello.jpg','images/hello.jpg','images/hello.jpg','images/hello.jpg','images/hello.jpg','images/hello.jpg'],
    #     locations = [   [0.0030, 0.3589, 1.0000, 1.0000],
    #                     [0.0779, 0.6744, 0.9768, 1.0000],
    #                     [0.2236, 0.0000, 0.7809, 0.4352],
    #                     [0.0000, 0.0000, 0.4313, 0.4505],
    #                     [0.6275, 0.1050, 0.9444, 0.2497],
    #                     [0.0000, 0.3859, 0.1250, 0.6922],
    #                     [0.7137, 0.2389, 0.8540, 0.4549],
    #                     [0.0000, 0.0000, 0.4667, 0.0630],
    #                     [0.3822, 0.4235, 0.4932, 0.6575],
    #                     [0.6616, 0.3617, 0.7880, 0.5165]  ],
    #     alpha_type = [0.0, 0, 1.0],  
    #     has_text_mask = 1,  
    #     has_image_mask = 0,  
    #     save_folder_name="pizza_1_0"
    # ),




]