### Google Slides pro tips for lightning talks
1. Automatic slide advance
    - *File > Publish to the web... > Auto-advance slides > Every 30 seconds*
    - uncheck the box *Start slideshow as soon as the player loads*
    - edit the generated URL to replace 30000 with 20000 (for 20 seconds)
2. Laser pointer without menu popup
    - go to full screen
    - start the presentation on the intro slide by clicking the play button
    - complete the next steps within 20 seconds before the slide advances
    - move mouse down to make menu appear, right-click on it, choose *Inspect*
    - select and delete the following element
      ```
      <div class="punch-viewer-nav-v2 punch-viewer-nav-floating \
      punch-viewer-nav-fade-out">
      ```
    - press L to toggle the laser pointer
    - to get the menu bar back, do Ctrl-R to refresh the web page (note: this will stop the auto-advance)
3. Extra slides
    - in film strip, right-click on slide, choose *Skip slide*
    - they won't show up as the slides auto-advance
    - but they can be seen if you navigate to them manually
