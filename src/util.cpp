
namespace svr_util {

template <typename Type>
inline Type square(Type x)
{
	return x * x; // will only work on types for which there is the * operator.
}

}
