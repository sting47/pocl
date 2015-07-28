/* author: sting47 */

void enqueue_kernel( void(^block)(void) )
{
  block();
}
